# Deep Reinforcement Learning in Pytorch

### Running

```bash
$ python3 main.py --semantic_encoder_model_filename ../semantic_encoding/models/20191011_201104_checkpoint_lr1e-05_30eps_att_proc3_washingtonramos.pth.tar --input_video_filename ../../../../datasets/Example/example.mp4 --user_document_filename ../semantic_encoding/resources/test_people_computer.txt --also_test --nb_episodes 1000 --gamma 1.0 --learning_rate 1e-7 --algo REINFORCE --eps 10000 --env FSE
```