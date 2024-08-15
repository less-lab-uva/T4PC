import torch

def get_val_violations(tqdm_loader, device, model, my_own_loss, args):
    data = {
        "img_id":[],
        "acc_label":[],
        "acc_prop_mask":[],
        "steer_label":[],
        "steer_prop_mask":[],
        "prop_name":[],
        "violation":[],
        "acc_pred":[],
        "steer_pred":[],
    }
            
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_loader):
            inputs = batch['front_img'].to(device)
            acceleration = batch['acceleration'].to(device)
            steering_angle = batch['steer'].to(device)

            outputs = model(inputs)
            
            acc_prop_mask = torch.ones(batch['acceleration'].shape[0], device=device)
            steer_prop_mask = torch.ones(batch['steer'].shape[0], device=device)

            # Property loss
            for i in range(len(batch["prop_eval"])):
                prop_name = batch["prop_name"][i][0]
                if prop_name in args.properties:
                    prop_attribute = "acceleration_continuous" if batch["prop_attribute"][i][0] == "acceleration" else "steering_angle"

                    # Mask v1: use wrong labels to mask main loss
                    if prop_attribute == "acceleration_continuous":
                        _,_,mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), acceleration, batch["prop_comparison"][i][0])
                        acc_prop_mask *= (~mask).long()
                    elif prop_attribute == "steering_angle":
                        _,_,mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), steering_angle, batch["prop_comparison"][i][0])
                        steer_prop_mask *= (~mask).long()
                    else:
                        raise NotImplementedError("No mask for property attribute")

                    prop_loss, vc, mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), outputs[prop_attribute], batch["prop_comparison"][i][0])
                    data["violation"].extend(mask.cpu().tolist())
                    data["img_id"].extend(list(batch['img_dir'][0]))
                    data["acc_label"].extend(batch['acceleration'].tolist())
                    data["acc_prop_mask"].extend(acc_prop_mask.tolist())
                    data["steer_label"].extend(batch['steer'].tolist())
                    data["steer_prop_mask"].extend(steer_prop_mask.tolist())
                    data["prop_name"].extend([prop_name] * len(mask))
                    data["acc_pred"].extend(outputs["acceleration_continuous"].cpu().tolist())
                    data["steer_pred"].extend(outputs["steering_angle"].cpu().tolist())

    return data