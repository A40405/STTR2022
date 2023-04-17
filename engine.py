
import os

from util.misc import MetricLogger, reduce_dict

from torchvision.utils import save_image

from datasets import denorm
import shutil


def save_batch_images(save_path,outputs,samples,style_images,epoch,it,device):
    
    outputs=denorm(outputs, device)
    samples.tensors=denorm(samples.tensors, device)
    style_images.tensors=denorm(style_images.tensors, device)
    save_image(outputs,os.path.join(save_path,"test_outputs",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  )  
    save_image(samples.tensors,os.path.join(save_path,"test_content_images",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  )  
    save_image(style_images.tensors,os.path.join(save_path,"test_style_images",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  ) 

def test_st(model, criterion, postprocessors, data_loader,  device, logger,epoch,save_path):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    print("len(data_loader):",len(data_loader))
    if os.path.exists(os.path.join(save_path,"test_outputs",f'{epoch:04}')):
        shutil.rmtree(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    os.makedirs(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    tmp_out=[]
    for it,(samples,style_images, targets) in metric_logger.log_every(data_loader, 100,logger, header):
#         if it<=3:
#             continue
        samples = samples.to(device)
        style_images = style_images.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples,style_images)  
        
    
        loss_dict = criterion(outputs, samples,style_images)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if it % 1==0:
            if not os.path.exists(os.path.join(save_path,"test_content_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_content_images",f'{epoch:04}'))
            if not os.path.exists(os.path.join(save_path,"test_style_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_style_images",f'{epoch:04}'))
                
          
            if isinstance(outputs, tuple):
                outputs,_=outputs
                
#             if False:
            if "content_image_name" in targets[0]:
                for i in range(len(outputs)):
                    c_name=targets[i]["content_image_name"]
                    s_name=targets[i]["style_image_name"]
                    save_name="{}_{}".format(c_name,s_name)
                    
                    output_i=denorm(outputs[i], device)
                    save_image(output_i,os.path.join(save_path,"test_outputs",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  )  
                    
                    sample_i=denorm(samples.tensors[i], device)
                    save_image(sample_i,os.path.join(save_path,"test_content_images",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  )  
                    
            else:
                save_batch_images(save_path,outputs,samples,style_images,epoch,it,device)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats
