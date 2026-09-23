import torch
import onnx
import onnxruntime as ort

from models.cnn import CNN

def main() -> None:
    model = CNN()

    state_dict = torch.load(
        '../artifacts/best_model.pth',
        map_location = 'cpu'
        weights_only = True
    )

    model.load_state_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(
        8, 1, 28, 28
    )

    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        input_names=['input'],
        output_names=['output'],
        dynamic_shapes={
            "x": {0: "batch_size"},
        },
        dynamo=True
    )

    onnx_program.save(
        'model_repository/cnn/1/model.onnx'
    )

    onnx_model = onnx.load(
        'model_repository/cnn/1/model.onnx'
    )

    # Validation
    onnx.checker.check_model(onnx_model)
    with torch.inference_mode():
        torch_output = model(dummy_input)

    session = ort.InferenceSession(
        'model_repository/cnn/1/model.onnx',
        providers=['CPUExecutionProvider']
    )

    onnx_output = session.run(
        ["output"], {'input': dummy_input.numpy()}
    )[0]

    torch.testing.assert_close(
        torch_output,
        torch.tensor(onnx_output),
        rtol=1e-3,
        atol=1e-5
    )

    print("ONNX Model export a validated.")

if __name__ == '__main__':
    main()