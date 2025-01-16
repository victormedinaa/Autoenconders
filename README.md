# Aumento de Resolución en Imágenes con Autoencoders

Este notebook utiliza autoencoders para aumentar la resolución de imágenes del conjunto de datos MNIST. El objetivo es reducir el tamaño de las imágenes a la mitad y luego reconstruirlas, mejorando así su resolución.

## Requisitos

- **PyTorch**
- **torchvision**
- **matplotlib**

Instala las dependencias con:

```bash
pip install torch torchvision matplotlib
```

## Descripción

1. **Preparación de Datos**: Carga y transforma el conjunto de datos MNIST a tensores.

```python
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)
```

2. **Autoencoder**: El modelo consta de un codificador (encoder) y un decodificador (decoder) para reducir y luego reconstruir las imágenes.

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(...)  # Codificador
        self.decoder = nn.Sequential(...)  # Decodificador
    def forward(self, x):
        return self.decoder(self.encoder(x))
```

3. **Entrenamiento**: Se realiza un *resize* de las imágenes a 14x14 y se optimiza el modelo utilizando el optimizador Adam.

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

4. **Evaluación**: Durante la prueba, se comparan las imágenes originales con las reconstruidas para observar la mejora en resolución.

```python
model.eval()
for i, img_original in enumerate(images[:3]):
    img_transformada = transformacion(img_original)
    pred = model(img_transformada)
    plt.imshow(pred.detach().numpy()[0])
```

## Resultados

El modelo muestra mejoras significativas en la calidad de las imágenes, con una disminución en la pérdida a lo largo de las épocas.

---

Este notebook es útil para explorar el uso de autoencoders en tareas de mejora de resolución de imágenes.
