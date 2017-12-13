/// Extracts THE MNIST DATABASE of handwritten digits as images (http://yann.lecun.com/exdb/mnist/) content as user human recognized images.
module MnistDatabaseExtraction

open System.IO
open SixLabors.ImageSharp
open MnistDatabase
open System

let negativeByte b = Byte.MaxValue - b
let toImage width height (data : byte[]) =
    let img = new Image<Rgba32>(width, height)
    for i = 0 to (height - 1) do
        for j = 0 to (width - 1) do
            let dark = data.[i * width + j] |> negativeByte
            let pixel = Rgba32(dark, dark, dark)
            img.[j, i] <- pixel
    img
let imageData (img : Image<Rgba32>) = 
    let pixelValue i j = 
        let p = img.[j, i]
        let a = List.averageBy (float) [p.R; p.G; p.B]
        let pixelValue = a |> byte |> negativeByte
        pixelValue
    Array.init ((int img.Height) * (int img.Width))  (fun p -> pixelValue (p / img.Width) (p % img.Width))
 
let saveImage width height (data : byte[]) fileName = 
    use img = toImage width height data
    use file = File.Create(fileName)
    img.SaveAsPng(file)
    ()

let rotateImage (img : Image<Rgba32>) (angleDegree : float32) =
    // After rotation image size may encrease and we will have to return image to its original size.
    let w1, h1 = img.Width, img.Height
    img.Mutate(fun x -> x.Rotate(angleDegree).BackgroundColor(Rgba32(Byte.MaxValue,Byte.MaxValue,Byte.MaxValue)) |> ignore)  
    let w2, h2 = img.Width, img.Height
    let left = (w2 - w1) / 2
    let top = (h2 - h1) / 2
    img.Mutate(fun x -> x.Crop(SixLabors.Primitives.Rectangle(left, top, w1, h1)) |> ignore)  
    assert (img.Width = w1 && img.Height = h1)
let rotateImageData angleDegree width height data  =
    use img = toImage width height data
    rotateImage img angleDegree
    imageData img

let saveImageByteLabeled destinationFolder (imageSize : ImageSize) (label, data) = 
    let imageFileName label = 
        let fileName = sprintf "%s.png" ((label).ToString())
        Path.Combine(destinationFolder, fileName)
    let imageBySize = saveImage imageSize.Width imageSize.Height
    imageBySize data (imageFileName label)