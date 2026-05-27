
| Stage | Experiment | train | models | dataset | multi_calssification |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 1     | baseline     | train     | models | dataset | multi_calssification |
| 1     | cnn     | cnn_train     | cnn_models | dataset | multi_calssification |
| 2     | 1DCNN     | efn_1dcnn_train     | efn_1dcnn_models | dataset | multi_calssification |
| 2     | XGboost     |      |  | dataset | multi_calssification |
| 3     | attention_based     | efn_mlp_atten_train     | efn_mlp_atten_models | dataset | efn_mlp_atten_multi_calssification |
| 3     | Produt_based     |   efn_mlp_product_train   | efn_mlp_product_models  | dataset | multi_calssification |
| 4     | focal_loss     |   efn_mlp_orig_focal_train   | models  | dataset | multi_calssification |
| 4     | Lwce_loss     |   efn_mlp_orig_lwce_train   | models  | dataset | multi_calssification |

