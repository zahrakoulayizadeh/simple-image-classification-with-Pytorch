import torch, torchvision
from PIL import Image
from torchvision import transforms
import PySimpleGUI as sg
import os
import io

def classification(filename):
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    result = []
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        res = [categories[top5_catid[i]]+"        ", "Probability : "+str("{:.4f}".format(round(top5_prob[i].item(), 4)))]
        result.append(res)
    return result



search_result = []
file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

sg.theme('DarkAmber')
layout = [
    [sg.Text("Enter your Image", font=("Myriad Pro", 14)),
     sg.Input(size=(35, 2), key="-FILE-"),
     sg.FileBrowse(file_types=file_types)
     ],
    [
        [sg.Button('Classify', size=(10, 1))]],
    [sg.Listbox(values=[],
                size=(60, 15),
                change_submits=True,
                bind_return_key=True,
                auto_size_text=True,
                key='_SEARCH_LISTBOX_', enable_events=True, horizontal_scroll=True),sg.Image(key="-IMAGE-",size=(400, 150))],

    [sg.Button('Exit', size=(15, 1))]
]

window = sg.Window('Image Classification', layout, margins=[0, 0], font=("Myriad Pro", 12), element_justification='l', finalize=True)

while True:
    event, values = window.read()

    if event == 'Classify':
        filename = values["-FILE-"]
        if os.path.exists(filename):
            image = Image.open(values["-FILE-"])
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE-"].update(data=bio.getvalue())
        search_result = [classification(filename)]
        # print(classification(values[0]))
        for item in search_result:
            window.Element('_SEARCH_LISTBOX_').update(item)
        search_result = []

    elif event == 'Back':
        window.close()

    # if user closes window or clicks on Exit
    elif event == sg.WIN_CLOSED or event == 'Exit':

        break

window.close()
