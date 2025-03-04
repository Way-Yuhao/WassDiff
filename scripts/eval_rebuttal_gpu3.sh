python ./src/eval.py trainer.devices=[3] name=sbdm_r trainer=gpu experiment=eval_val_set_sbdm_rebuttal data.stop_at_batch=40;
python ./src/eval.py trainer.devices=[3] name=sbdm_r_2 trainer=gpu experiment=eval_val_set_sbdm_rebuttal data.stop_at_batch=40;
python ./src/eval.py trainer.devices=[3] name=sbdm_r_3 trainer=gpu experiment=eval_val_set_sbdm_rebuttal data.stop_at_batch=40;
python ./src/eval.py trainer.devices=[3] name=sbdm_r_4 trainer=gpu experiment=eval_val_set_sbdm_rebuttal data.stop_at_batch=40;
python ./src/eval.py trainer.devices=[3] name=sbdm_r_5 trainer=gpu experiment=eval_val_set_sbdm_rebuttal data.stop_at_batch=40;
#python ./src/eval.py trainer.devices=[3] name=sbdm_r_6 trainer=gpu experiment=eval_val_set_sbdm_rebuttal data.stop_at_batch=40;
