import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.fairseq_nat_model import FairseqNATDecoder, ensemble_decoder
from fairseq.models.transformer import Embedding
from ..torch_imputer import best_alignment
from collections import Counter
from CTCS2UT.models.s2t_conformer_nat import S2TConformerNATModel

logger = logging.getLogger(__name__)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    bsz = src_lens.size(0)
    ratio = int(max_trg_len / max_src_len)
    index_t = utils.new_arange(trg_lens, max_src_len)
    index_t = torch.repeat_interleave(index_t, repeats=ratio, dim=-1).unsqueeze(0).expand(bsz, -1)
    return index_t 


@register_model("s2t_conformer_ctc_ngram_glat")
class S2TConformerCtcGlatModel(S2TConformerNATModel):
    """
    Non-autoregressive Conformer-based Speech-to-text Model.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.plain_ctc = args.plain_ctc
        if not self.plain_ctc:
            from ctcdecode import CTCBeamDecoder
            import multiprocessing
            self.ctc_decoder = CTCBeamDecoder(
                decoder.dictionary.symbols,
                model_path=None,
                alpha=0,
                beta=0,
                cutoff_top_n=40,
                cutoff_prob=1.0,
                beam_width=args.ctc_beam_size,
                num_processes=multiprocessing.cpu_count(),
                blank_id=decoder.dictionary.blank_index,
                log_probs_input=True
            )
        self.src_upsample_ratio = args.src_upsample_ratio

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        S2TConformerNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            '--ctc-beam-size',
            type=int
        )
        parser.add_argument(
            '--decoder-learned-pos', 
            action='store_true', 
            help='use learned positional embeddings in the decoder'
        )
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        args.tgt_dict_size = len(task.target_dictionary)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        base_model = cls(args, encoder, decoder)

        return base_model

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = NATransformerDecoder(args, task.tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def sequence_ngram_loss_with_logits(self, logits, logit_mask, targets):
        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        
        if not self.args.sctc_loss:
            if self.args.ngram_size == 1:
                loss = self.compute_ctc_1gram_loss(log_probs, logit_mask, targets)
            elif self.args.ngram_size == 2:
                loss = self.compute_ctc_bigram_loss(log_probs, logit_mask, targets)
            else:
                raise NotImplementedError
        else:
            loss = self.compute_sctc_ngram_loss(log_probs, targets, self.args.ngram_size)
        
        return loss
    
    def compute_ctc_1gram_loss(self, log_probs, logit_mask, targets):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        bow = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        bow[:,self.tgt_dict.blank_index] = 0
        ref_bow = torch.zeros(batch_size, vocab_size).cuda(probs.get_device())
        ones = torch.ones(batch_size, vocab_size).cuda(probs.get_device())
        ref_bow.scatter_add_(-1, targets, ones).detach()
        ref_bow[:,self.pad] = 0
        expected_length = torch.sum(bow).div(batch_size)
        loss = torch.mean(torch.norm(bow-ref_bow,p=1,dim=-1))/ (length_tgt + expected_length)
        return loss

    def compute_ctc_bigram_loss(self, log_probs, logit_mask, targets):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        targets = targets.tolist()
        probs_blank = probs[:,:,self.tgt_dict.blank_index]
        length = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        length[:,self.tgt_dict.blank_index] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,self.tgt_dict.blank_index]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_1 = []
        gram_2 = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - 1
        for i in range(batch_size):
            two_grams = Counter()
            gram_1.append([])
            gram_2.append([])
            gram_count.append([])
            for j in range(num_grams):
                two_grams[(targets[i][j], targets[i][j+1])] += 1
            j = 0
            for two_gram in two_grams:
                if self.pad in two_gram:
                    continue
                gram_1[-1].append(two_gram[0])
                gram_2[-1].append(two_gram[1])
                gram_count[-1].append(two_grams[two_gram])
                if two_gram[0] == two_gram[1]:
                    rep_gram_pos.append((i, j))
                j += 1
            while len(gram_count[-1]) < num_grams:
                gram_1[-1].append(1)
                gram_2[-1].append(1)
                gram_count[-1].append(0)
        gram_1 = torch.LongTensor(gram_1).cuda(blank_matrix.get_device())
        gram_2 = torch.LongTensor(gram_2).cuda(blank_matrix.get_device())
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        gram_1_probs = torch.gather(probs, -1, gram_1.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, length_ctc, 1)
        gram_2_probs = torch.gather(probs, -1, gram_2.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
        probs_matrix = torch.matmul(gram_1_probs, gram_2_probs)
        bag_grams = blank_matrix.view(batch_size, 1, length_ctc, length_ctc) * probs_matrix
        bag_grams = torch.sum(bag_grams.view(batch_size, num_grams, -1), dim = -1).view(batch_size, num_grams,1)
        if len(rep_gram_pos) > 0:
            for pos in rep_gram_pos:
                i, j = pos
                gram_id = gram_1[i, j]
                gram_prob = probs[i, :, gram_id]
                rep_gram_prob = torch.sum(gram_prob[1:] * gram_prob[:-1])
                bag_grams[i, j, 0] = bag_grams[i, j, 0] - rep_gram_prob
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)

        assert match_gram <= length_tgt - 1
        assert match_gram <= expected_length - 1

        loss = (- 2 * match_gram).div(length_tgt + expected_length - 2)
        
        return loss

    def compute_sctc_ngram_loss(self, log_probs, logit_mask, targets, n):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        targets = targets.tolist()
        probs_blank = probs[:,:,self.tgt_dict.blank_index]
        length = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        length[:,self.tgt_dict.blank_index] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,self.tgt_dict.blank_index]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_idx = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - n + 1
        for i in range(batch_size):
            ngrams = Counter()
            gram_idx.append([])
            gram_count.append([])
            for j in range(num_grams):
                idx = []
                for k in range(n):
                    idx.append(targets[i][j+k])
                idx = tuple(idx)
                ngrams[idx] += 1

            for k in range(n):
                gram_idx[-1].append([])
            for ngram in ngrams:
                for k in range(n):
                    gram_idx[-1][k].append(ngram[k])
                gram_count[-1].append(ngrams[ngram])

            while len(gram_count[-1]) < num_grams:
                for k in range(n):
                    gram_idx[-1][k].append(1)
                gram_count[-1].append(0)

        gram_idx = torch.LongTensor(gram_idx).cuda(blank_matrix.get_device()).transpose(0,1)
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        blank_matrix = blank_matrix.view(batch_size, 1, length_ctc, length_ctc)
        for k in range(n):
            gram_k_probs = torch.gather(probs, -1, gram_idx[k].view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
            if k == 0:
                state = gram_k_probs
            else:
                state = torch.matmul(state, blank_matrix) * gram_k_probs
        bag_grams = torch.sum(state, dim=-1)
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)
        
        assert match_gram <= length_tgt - (n-1)
        assert match_gram <= expected_length - (n-1)
        
        loss = (- 2 * match_gram).div(length_tgt + expected_length - 2*(n - 1))
        return loss

    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0,
                                      reduce=True
                                      ) -> torch.FloatTensor:
        # lengths : (batch_size, )
        # calculated by counting number of mask
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        targets = targets[target_mask.bool()]
        if reduce:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="none",
                zero_infinity=True,
            )
            loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, reduce=True, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        prev_output_tokens = self.initialize_output_tokens_by_upsampling(encoder_out)
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        output_length = prev_output_tokens_mask.sum(dim=-1)

        target_mask = tgt_tokens.ne(self.pad)
        target_length = target_mask.sum(dim=-1) 
        # glat_implemented_here
        glat_info = None
        oracle = None
        keep_word_mask = None
        
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                normalized_logits = self.decoder(
                    normalize=True,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )

                normalized_logits_T = normalized_logits.transpose(0, 1).float() #T * B * C, float for FP16

                best_aligns = best_alignment(normalized_logits_T, tgt_tokens, output_length, target_length, self.tgt_dict.blank_index, zero_infinity=True)
                #pad those positions with <blank>
                padded_best_aligns = torch.tensor([a + [0] * (normalized_logits_T.size(0) - len(a)) for a in best_aligns], device=prev_output_tokens.device, dtype=prev_output_tokens.dtype)
                oracle_pos = (padded_best_aligns // 2).clip(max=tgt_tokens.size(-1)-1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                oracle = oracle.masked_fill(padded_best_aligns % 2 == 0, self.tgt_dict.blank_index)
                oracle = oracle.masked_fill(~prev_output_tokens_mask, self.pad)
                
                _,pred_tokens = normalized_logits.max(-1)
                same_num = ((pred_tokens == oracle) & prev_output_tokens_mask).sum(dim=-1)
                keep_prob = ((output_length - same_num) / output_length * glat['context_p']).unsqueeze(-1) * prev_output_tokens_mask.float()

                keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()
        
                glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)

                glat_info = {
                    "glat_acc": (same_num.sum() / output_length.sum()).detach(),
                    "glat_context_p": glat['context_p'],
                    "glat_keep": keep_prob.mean().detach(),
                }
                prev_output_tokens = glat_prev_output_tokens
                
        
        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        
        if self.args.use_ngram:
            ctc_loss = self.sequence_ngram_loss_with_logits(word_ins_out, prev_output_tokens_mask, tgt_tokens)
        else:    
            ctc_loss = self.sequence_ctc_loss_with_logits(
                logits=word_ins_out,
                logit_mask=prev_output_tokens_mask,
                targets=tgt_tokens,
                target_mask=target_mask,
                blank_index=self.tgt_dict.blank_index,
                label_smoothing=self.args.label_smoothing,
                reduce=reduce
            )

        ret_val = {
            "ctc_loss": {"loss": ctc_loss},
        }    
        
        return ret_val, glat_info

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):        
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_lengths = (output_masks.bool()).long().sum(-1) 
        output_logits = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )

        if self.plain_ctc:    
            _scores, _tokens = output_logits.max(-1)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            def _ctc_postprocess(tokens):
                _toks = tokens.int().tolist()
                deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
                hyp = [v for v in deduplicated_toks if (v != self.tgt_dict.blank_index) and (v!= self.tgt_dict.pad_index)]
                return hyp
            
            
            # unpad_output_tokens = []
            # for output_token in output_tokens:
            #     unpad_output_tokens.append(_ctc_postprocess(output_token))

            # res_lengths = torch.tensor([len(res) for res in unpad_output_tokens],device=decoder_out.output_tokens.device, dtype=torch.long)
            # res_seqlen = max(res_lengths.tolist())
            # res_tokens = [res + [self.tgt_dict.pad_index] * (res_seqlen - len(res)) for res in unpad_output_tokens]
            # res_tokens = torch.tensor(res_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)

            # output_nll = - _scores.masked_fill(~output_masks, 0).sum(dim=-1)
            # output_trans_nll = output_nll
            # output_logpy = F.ctc_loss(
            #     output_logits.transpose(0, 1).float(),  # compatible with fp16 T * B * C
            #     res_tokens,
            #     output_lengths,
            #     res_lengths,
            #     blank=self.tgt_dict.blank_index,
            #     reduction="none",
            #     zero_infinity=True,
            # )

            # assert (output_nll >= output_logpy).all()
 
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=None,
                history=history,
            )
        else:
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(output_logits)
            top_beam_tokens = beam_results[:, 0, :]
            top_beam_len = out_lens[:, 0]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
            
            if history is not None:
                history.append(output_tokens.clone())

            output_nll = beam_scores[:,0]
            output_trans_nll = output_logpy = output_nll
            return decoder_out._replace(
                output_tokens=top_beam_tokens.to(output_logits.device),
                output_scores=torch.full(top_beam_tokens.size(), 1.0),
                attn=None,
                history=history,
            )
    
    def initialize_output_tokens_by_upsampling(self, encoder_out):
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_tokens = encoder_out["encoder_padding_mask"][0]
        else:
            T, B, _ = encoder_out["encoder_out"][0].shape
            src_tokens = torch.zeros(B, T).bool().to(encoder_out["encoder_out"][0].device)
        src_tokens = src_tokens.long()

        if self.src_upsample_ratio <= 1:
            return src_tokens

        def _us(x, s):
            B = x.size(0)
            _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
            return _x
        
        return _us(src_tokens, self.src_upsample_ratio)
        
    
    def initialize_output_tokens(self, encoder_out, src_tokens, src_lengths):
        initial_output_tokens = self.initialize_output_tokens_by_upsampling(encoder_out)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )
    
    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.input_proj = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim)
        self.encoder_embed_dim = args.encoder_embed_dim
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, oracle=None, keep_word_mask=None, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        oracle=None,
        keep_word_mask=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_out"][0].detach().transpose(0, 1)
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(src_embd.size(0), src_embd.size(1)).bool()
            )

            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )
            if oracle is not None:
                oracle_embedding, _ = self.forward_embedding(oracle)
                x = x.masked_fill(keep_word_mask.unsqueeze(-1), 0) + oracle_embedding.masked_fill(~keep_word_mask.unsqueeze(-1), 0)
                

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = self.input_proj(states)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding


@register_model_architecture(
    "s2t_conformer_ctc_ngram_glat", "s2t_conformer_ctc_ngram_glat"
)
def base_architecture(args):
    # conformer args
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.input_channels = getattr(args, "input_channels", 1)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # for Conv1d
    args.conv_channels = getattr(args, "conv_channels", 1024)  # for Conv1d
    args.conv_out_channels = getattr(args, "conv_out_channels", 256)  # for Conv2d
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    
    # nat args
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.src_upsample_ratio = getattr(args, "src_upsample_ratio", 1)
    args.plain_ctc = getattr(args, "plain_ctc", False)
    args.ctc_beam_size = getattr(args, "ctc_beam_size", 20)