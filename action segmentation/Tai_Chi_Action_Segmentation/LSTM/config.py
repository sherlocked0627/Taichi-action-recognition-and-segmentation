import argparse

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode',  type=str, default='train')
	# parser.add_argument('--train_path', type=str, default='./dataset/train.pkl')
	parser.add_argument('--train_path', type=str, default='./dataset/TaiChi/train/data.npy')
	parser.add_argument('--valid_path', type=str, default='./dataset/TaiChi/valid/data.npy')
	parser.add_argument('--test_path',  type=str, default='./dataset/TaiChi/test/data.npy')
	parser.add_argument('--train_label_path',  type=str, default='./dataset/TaiChi/train/label.npy')
	parser.add_argument('--valid_label_path',  type=str, default='./dataset/TaiChi/valid/label.npy')
	parser.add_argument('--test_label_path',  type=str, default='./dataset/TaiChi/test/label.npy')
	parser.add_argument('--model_path',  type=str, default='./models')

	parser.add_argument('--batch_size',  type=int, default=1530)
	parser.add_argument('--learning_rate',type=int, default=0.005)
	parser.add_argument('--beta1',type=int, default=0.5)
	parser.add_argument('--beta2',type=int, default=0.99)
	parser.add_argument('--dropout',type=int, default=0)
	parser.add_argument('--weight_decay',type=int, default=0)

	parser.add_argument('--num_epochs',type=int, default=100)
	parser.add_argument('--start_epoch',type=int, default=0)
	parser.add_argument('--test_epoch',type=int, default=100)
	parser.add_argument('--val_step',type=int, default=1)

	parser.add_argument('--num_classes',type=int, default=10)
	parser.add_argument('--embed_dim',type=int, default=51)

	parser.add_argument('--lstm_hidden_dim',type=int, default=64)
	parser.add_argument('--lstm_num_layers',type=int, default=5)
	

	args = parser.parse_args()

	return args
