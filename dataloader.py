from PIL import Image
import os
import csv
import torch
from torch.utils import data
from torchvision import transforms

def prepare_dataset(opts):
	
	train_op = os.path.join(opts.data_folder,"A")
	test_op = os.path.join(opts.data_folder,"B")
	train_set = load_ops(train_op,opts)
	test_set = load_ops(test_op,opts)

	return train_set, test_set

# creates a separate dataloader for each operation (OP)
def load_ops(folder,opts):
	ops = []
	for ID in os.listdir(folder):
		op_path = folder+"/"+ID
		if os.path.isdir(op_path):
			anno_file = os.path.join(opts.annotation_folder,"video" + ID + "-tool.txt")
			dataset = Cholec80Anticipation(op_path, anno_file, opts.horizon)
			dataloader = data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)
			ops.append((ID,dataloader))
	return ops

# generates the ground truth signal over time for a single tool and a single operation
def generate_anticipation_gt_onetool(tool_code,horizon):
	# initialize ground truth signal
	anticipation = torch.zeros_like(tool_code).type(torch.FloatTensor)
	# default ground truth value is <horizon> minutes
	# (i.e. tool will not appear within next <horizon> minutes)
	anticipation_count = horizon
	# iterate through tool-presence signal backwards
	for i in torch.arange(len(tool_code)-1,-1,-1):
		# if tool is present, then set anticipation value to 0 minutes
		if tool_code[i]:
			anticipation_count = 0
		# else increase anticipation value with each (reverse) time step but clip at <horizon> minutes
		# video is sampled at 1fps, so 1 step = 1/60 minutes
		else:
			anticipation_count = min(horizon, anticipation_count + 1/60)
		anticipation[i] = anticipation_count
	# normalize ground truth signal to values between 0 and 1
	anticipation = anticipation / horizon
	return anticipation

# generates the ground truth signal over time for a single operation
def generate_anticipation_gt(tools,horizon):
	return torch.stack([generate_anticipation_gt_onetool(tool_code,horizon) for tool_code in tools]).permute(1,0)

class Cholec80Anticipation(data.Dataset):
	def __init__(self, image_path, annotation_path, horizon=5):
		self.image_path = image_path
		
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		with open(annotation_path, "r") as f:
			tool_presence = []
			reader = csv.reader(f, delimiter='\t')
			next(reader, None)
			for i,row in enumerate(reader):
				if i == 0:
					self.offset = int(int(row[0])/25)
				tool_presence.append([int(row[x]) for x in [2,4,5,6,7]])
				# tool_presence.append([int(row[x]) for x in [2]])
			tool_presence = torch.LongTensor(tool_presence).permute(1,0)

		self.target_reg = generate_anticipation_gt(tool_presence,horizon)
		self.target_cls = torch.where((self.target_reg < 1) & (self.target_reg > 0),torch.Tensor([2]),self.target_reg).type(torch.long)

	def __getitem__(self, index):
		target_reg = self.target_reg[index]
		target_cls = self.target_cls[index]

		frame = self.image_path + "/frame%08d.jpg" % (index+self.offset)
		img = Image.open(frame)
		img = self.transform(img)

		return img, target_reg, target_cls

	def __len__(self):
		return len(self.target_reg)