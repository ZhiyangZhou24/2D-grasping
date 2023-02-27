import torch
import onnx
 
# gloabl variable
model_path = "logs/cornell_ftn/230227_2018_dwc1_rgbd_near_drop1_ga0_gama/epoch_42_iou_0.9859"
import sys
sys.path.append('/home/lab/zzy/grasp/2D-grasping-my')
if __name__ == "__main__":
    	# input shape尽量选择能被2整除的输入大小
	dummy_input = torch.randn(1, 4, 224, 224, device="cuda")
	# [1] create network
	# model = GenerativeResnet(input_channels=4, dropout=1, prob=0.1, channel_size=32)
	# model = model.cuda()
	# print("create grcnn model finised ...")          
	# [2] 加载权重
	model = torch.load(model_path)
    # model.load_state_dict(state_dict)
	# print("load weight to model finished ...")

	# convert torch format to onnx
	input_names = ["input"]
	output_names = ["output"]
	torch.onnx.export(model, 
		dummy_input, 
		"onnx_files/dsc_rdbd_2.onnx", 
		verbose=True, 
		input_names=input_names,
		output_names=output_names)
	print("convert torch format model to onnx ...")
	# [4] confirm the onnx file
	net = onnx.load("onnx_files/dsc_rdbd_2.onnx")
	# check that the IR is well formed
	onnx.checker.check_model(net)
	# print a human readable representation of the graph
	onnx.helper.printable_graph(net.graph)