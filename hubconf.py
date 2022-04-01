dependencies = ['torch', 'torchvision', 'torchvideo', 'numpy']

from text_driven_video_acceleration import JointModel

def TextDrivenVideoAcceleration(pretrained=False, progress=False, sent_emb_size=512, hidden_feat_emb_size=512, final_feat_emb_size=128, sent_att_size=1024, word_att_size=1024, use_visual_shortcut=True, use_sentence_level_attention=True, use_word_level_attention=True, learn_first_hidden_vector=False, action_size=3):
    
    model = JointModel(vocab_size=400002, # Number of words in the GloVe vocabulary
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
                       action_size=action_size,
                       pretrained=pretrained)

    return model