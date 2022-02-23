from SoccerNet.Evaluation.ActionSpotting import evaluate

results = evaluate(SoccerNet_path='/data-net/datasets/SoccerNetv2/ResNET_TF2', 
                   Predictions_path='/home-net/axesparraguera/data/test_predictions',
                   split="test", version=2, prediction_file="results_spotting.json")

print("Average mAP: ", results["a_mAP"])
print("Average mAP per class: ", results["a_mAP_per_class"])
print("Average mAP visible: ", results["a_mAP_visible"])
print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("Average mAP unshown: ", results["a_mAP_unshown"])
print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])