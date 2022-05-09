import torch
from dataloader import prepare_dataset
from model import TimeModel
from options_traintest import parser
import time

opts = parser.parse_args()
train_set, test_set = prepare_dataset(opts)
model = TimeModel(opts)
start_time = time.time()

with open(model.log_path, "w") as log_file:

	for epoch in range(1,opts.epochs+1):

		model.reset_stats()
		model.net.train()

		for _,op in train_set:

			model.init_op('train')

			for batch,(data,target_reg,target_cls, t_elapsed) in enumerate(op):
				t_elapsed = t_elapsed.to(torch.float32)
				data, target_reg, target_cls, t_elapsed = data.cuda(), target_reg.cuda(), target_cls.cuda(), t_elapsed.cuda()

				output_reg, output_cls = model.forward(data,t_elapsed)
				loss = model.compute_loss(output_reg,output_cls,target_reg,target_cls)
				model.backward(loss,batch)

				if model.optimized:
					h, c = hidden_state
					hidden_state = (h.detach(),c.detach())

				model.update_stats(
					loss.item(),
					output_reg.detach(),
					output_cls.detach(),
					target_reg.detach(),
					target_cls.detach(),
					mode='train'
				)

			if not model.optimized:
				model.optimizer.step()

		with torch.no_grad():

			model.net.eval()

			for ID,op in test_set:

				model.init_op('test')

				for data,target_reg,target_cls,t_elapsed in op:
					t_elapsed = t_elapsed.to(torch.float32)
					data, target_reg, target_cls = data.cuda(), target_reg.cuda(), target_cls.cuda(), t_elapsed.cuda()

					output_reg, output_cls, hidden_state = model.forward(data,hidden_state)
					loss = model.compute_loss(output_reg,output_cls,target_reg,target_cls)

					model.update_stats(
						loss.item(),
						output_reg.detach(),
						output_cls.detach(),
						target_reg.detach(),
						target_cls.detach(),
						mode='test'
					)

		model.summary(log_file,epoch)        

print(time.time()-start_time)          
            
