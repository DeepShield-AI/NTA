### Environment

- flowpic

### About Classification

- 2class: Novpn, Vpn
- 6class: Chat, Email, File, P2p, Streaming, Voip
- 12class: Chat, Email, File, P2p, Streaming, Voip, Vpn_Chat, Vpn_Email, Vpn_File, Vpn_P2p, Vpn_Streaming, Vpn_Voip

### About Data

- Data is directly used by the author of the paper

### Usage 
- Run `1d_cnn/cnn_1d_torch` for 12-classification
- Modify lines 31-34 to change the dataset
- Modify line 26 of `label_num` variable and line 38 to change the task

### About Original Code
`1d_cnn/cnn_1d_tensorflow` is the original code, which has not been debugged, and is provided for reference

Original code:
[https://github.com/mydre/wang-wei-s-research](https://github.com/mydre/wang-wei-s-research)