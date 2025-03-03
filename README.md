# Image Resolution Enhancement with Autoencoders

This notebook leverages autoencoders to boost the resolution of images from the MNIST dataset. The idea is to first shrink the images to half their original size and then reconstruct them, effectively enhancing their resolution.

## Requirements

- **PyTorch**
- **torchvision**
- **matplotlib**

Install the dependencies with:

```bash
pip install torch torchvision matplotlib
```

## Description

1. **Data Preparation**: Load and transform the MNIST dataset into tensors.

   ```python
   transform = transforms.ToTensor()
   mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)
   ```

2. **Autoencoder**: The model is built with an encoder and a decoder that compress and then reconstruct the images.

   ```python
   class Autoencoder(nn.Module):
       def __init__(self):
           super().__init__()
           self.encoder = nn.Sequential(...)  # Encoder
           self.decoder = nn.Sequential(...)  # Decoder
       def forward(self, x):
           return self.decoder(self.encoder(x))
   ```

3. **Training**: The images are resized to 14x14 pixels and the model is optimized using the Adam optimizer.

   ```python
   transformacion = transforms.Resize((14, 14))
   for epoch in range(num_epochs):
       for (img_original, _) in data_loader:
           img_transformada = transformacion(img_original)
           recon = model(img_transformada)
           loss = criterion(recon, img_original)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

4. **Evaluation**: During testing, the original images are compared with the reconstructed ones to see the improvement in resolution.

   ```python
   model.eval()
   for i, img_original in enumerate(images[:3]):
       img_transformada = transformacion(img_original)
       pred = model(img_transformada)
       plt.imshow(pred.detach().numpy()[0])
   ```

## Results

The model demonstrates significant improvements in image quality, with the loss steadily decreasing over the training epochs.

---

This notebook is a great resource for exploring how autoencoders can be used to enhance image resolution.
