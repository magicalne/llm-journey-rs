## Pytorch cheatsheet for Candle

<document>
Pytorch cheatsheet
Cheatsheet:

Using PyTorch	Using Candle
Creation	torch.Tensor([[1, 2], [3, 4]])	Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?
Creation	torch.zeros((2, 2))	Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?
Indexing	tensor[:, :4]	tensor.i((.., ..4))?
Operations	tensor.view((2, 2))	tensor.reshape((2, 2))?
Operations	a.matmul(b)	a.matmul(&b)?
Arithmetic	a + b	&a + &b
Device	tensor.to(device="cuda")	tensor.to_device(&Device::new_cuda(0)?)?
Dtype	tensor.to(dtype=torch.float16)	tensor.to_dtype(&DType::F16)?
Saving	torch.save({"A": A}, "model.bin")	candle::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?
Loading	weights = torch.load("model.bin")	candle::safetensors::load("model.safetensors", &device)
</document>

## Translate the python code into rust

You should replace torch with candle.

<code>
{code}
</code>
