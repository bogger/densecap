require 'torch'
require 'nn'

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'

local utils = require 'densecap.utils'
local eval_utils = require 'eval.eval_utils'

--[[
Evaluate a trained DenseCap model by running it on a split on the data.
--]]

local cmd = torch.CmdLine()
cmd:option('-data_h5', '', 'The HDF5 file to load data from; optional.')
cmd:option('-data_json', '', 'The JSON file to load data from; optional.')
cmd:option('-res_json', '', 'The JSON file to load results from.')
cmd:option('-gpu', 0, 'The GPU to use; set to -1 for CPU')
cmd:option('-use_cudnn', 1, 'Whether to use cuDNN backend in GPU mode.')
cmd:option('-split', 'val', 'Which split to evaluate; either val or test.')
cmd:option('-max_images', -1, 'How many images to evaluate; -1 for whole split')
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 1000)
cmd:option('-vis', 0,
  'if 1 then writes files needed for pretty vis into vis/ ')
cmd:option('-output_vis_dir', 'vis/data')
local opt = cmd:parse(arg)

  

local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
print(string.format('Using dtype "%s"', dtype))


-- Set up the DataLoader; use HDF5 and JSON files from checkpoint if they were
-- not explicitly provided.
local loader = DataLoader(opt)

-- Actually run evaluation
local eval_kwargs = {
  loader=loader,
  split=opt.split,
  max_images=opt.max_images,
  dtype=dtype,
  res_json=opt.res_json,
  vis=opt.vis,
  output_vis_dir=opt.output_vis_dir
}
print(eval_kwargs.res_json)
local eval_results = eval_utils.eval_split_from_file(eval_kwargs)
