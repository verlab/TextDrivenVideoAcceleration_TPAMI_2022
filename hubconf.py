dependencies = ['torch', 'torchvision', 'torchvideo', 'numpy']

from semantic_encoding.utils import init_weights
from text_driven_video_acceleration import JointModel
import torch
import numpy as np

def TextDrivenVideoAcceleration(pretrained=False, progress=False, sent_emb_size=512, hidden_feat_emb_size=512, final_feat_emb_size=128, sent_att_size=1024, word_att_size=1024, use_visual_shortcut=True, use_sentence_level_attention=True, use_word_level_attention=True, word_embeddings=None, fine_tune_word_embeddings=False, fine_tune_resnet=False, learn_first_hidden_vector=False, action_size=3, ):
    if not word_embeddings:
        word_embeddings = np.random.random((400002, 300)).astype(np.float32)

    model = JointModel(vocab_size=400002,
                       doc_emb_size=512,  # R(2+1)D embedding size
                       sent_emb_size=sent_emb_size,
                       word_emb_size=300,  # GloVe embeddings size
                       sent_rnn_layers=1,
                       word_rnn_layers=1,
                       hidden_feat_emb_size=hidden_feat_emb_size,
                       final_feat_emb_size=final_feat_emb_size,
                       sent_att_size=sent_att_size,
                       word_att_size=word_att_size,
                       use_visual_shortcut=use_visual_shortcut,
                       use_sentence_level_attention=use_sentence_level_attention,
                       use_word_level_attention=use_word_level_attention,
                       learn_first_hidden_vector=learn_first_hidden_vector,
                       action_size=action_size)

    # Init word embeddings layer with pretrained embeddings
    model.vdan_plus.text_embedder.doc_embedder.sent_embedder.init_pretrained_embeddings(torch.from_numpy(word_embeddings))
    model.vdan_plus.text_embedder.doc_embedder.sent_embedder.allow_word_embeddings_finetunening(fine_tune_word_embeddings)  # Make it available to finetune the word embeddings
    model.vdan_plus.vid_embedder.fine_tune(fine_tune_resnet)  # Freeze/Unfreeze R(2+1)D layers. We didn't use it in our paper. But, feel free to try ;)

    model.vdan_plus.apply(init_weights)  # Apply function "init_weights" to all FC layers of our model.

    if pretrained:
        vdan_plus_state_dict = torch.hub.load_state_dict_from_url('https://github.com/verlab/TextDrivenVideoAcceleration_TPAMI_2022/releases/download/pre_release/vdan+_pretrained_model.pth', progress=progress)
        agent_state_dict = torch.hub.load_state_dict_from_url('https://github.com/verlab/TextDrivenVideoAcceleration_TPAMI_2022/releases/download/pre_release/youcookii_saffa_model.pth', progress=progress)

        model.vdan_plus.load_state_dict(vdan_plus_state_dict['model_state_dict'])
        model.policy.load_state_dict(agent_state_dict['policy_state_dict'])

    return model