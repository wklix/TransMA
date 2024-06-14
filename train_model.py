from train import MolTrain
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=100,
                learning_rate=0.00001,
                batch_size=4,
                early_stopping=5,
                save_path='./new_data',
                remove_hs=True,
              )
clf.fit('./dataset/RaW/cliff/train.csv') 