// THE MNIST DATABASE reading module
module MnistDatabase

open System
open System.IO
open System.IO.Compression

let openUncompressedStream fileName : Stream =
    let fileStream = File.OpenRead(fileName)
    new GZipStream(fileStream, CompressionMode.Decompress) :> Stream

let readNetworkInt (binaryReader : BinaryReader) = 
    let intRaw = binaryReader.ReadInt32()
    // According to THE MNIST DATABASE documentation (http://yann.lecun.com/exdb/mnist/)
    // All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. 
    System.Net.IPAddress.NetworkToHostOrder(intRaw)

type ImageSize = { Height : int; Width : int }
type ImagesFileHeader = { ImagesCount : int; Size : ImageSize }

let mnistImagesHeader stream =
    let binaryReader = new BinaryReader(stream)
    let readNextInt () = readNetworkInt binaryReader
    let magicNumber = readNextInt ()
    if magicNumber <> 2051 then raise (Exception("Magic number expected."))
    let imagesNumber = readNextInt ()
    let rowsNumber = readNextInt ()
    let columnsNumber = readNextInt ()
    { ImagesCount = imagesNumber; Size = { Height = rowsNumber; Width = columnsNumber }}

let mnistImagesSeq header stream = 
    let binaryReader = new BinaryReader(stream)
    let imageByteSize = header.Size.Height * header.Size.Width
    seq { for __ = 1 to header.ImagesCount do yield binaryReader.ReadBytes(imageByteSize)}

let mnistImagesData stream = 
    let header = mnistImagesHeader stream
    let sequence = mnistImagesSeq header stream
    header, sequence

let mnistLabelsHeader stream =
    let binaryReader = new BinaryReader(stream)
    let readNextInt () = readNetworkInt binaryReader
    let magicNumber = readNextInt ()
    if magicNumber <> 2049 then raise (Exception("Magic number expected."))
    readNextInt ()

let mnistLabelsSeq itemsCount stream = 
    let binaryReader = new BinaryReader(stream)
    seq { for __ = 1 to itemsCount do yield binaryReader.ReadByte() }

let mnistLabelsData stream : (int * byte seq) = 
    let itemsCount = mnistLabelsHeader stream
    let sequence = mnistLabelsSeq itemsCount stream
    itemsCount, sequence

// THE MNIST DATABASE of handwritten digits (http://yann.lecun.com/exdb/mnist/) are stored in pairs of custom binary files. 
// Images file contains array of image data.
// Labels file contains number digits depicted on images.
type MnistDataFileNamesPair = { Labels: string; Images: string }

let extractFile compressedFileName =
    use source = File.OpenRead(compressedFileName)
    let destinationFileName = Path.GetTempFileName()
    let destination = File.Create(destinationFileName)
    let gzipStream = new GZipStream(source, CompressionMode.Decompress, true) 
    gzipStream.CopyTo(destination)
    destination.Position <- 0L
    let removeFile = { 
        new IDisposable with 
            member __.Dispose() = 
                destination.Dispose()
                if (File.Exists(destinationFileName)) then File.Delete(destinationFileName)
    }
    destination, removeFile

let disposableComposite (disposables : IDisposable list) = { 
    new IDisposable with 
        member __.Dispose() = disposables |> List.iter (fun d -> d.Dispose())
}

// Data files may be opened once but read all sequence multiple times.
let mnistLabeledImageDataExt (dataFiles : MnistDataFileNamesPair) =
    let imagesStream, imageStorage = extractFile dataFiles.Images
    try
        let labelsStream, labelsStorage = extractFile dataFiles.Labels
        try
            let imagesHeader = mnistImagesHeader imagesStream
            let labelsCount = mnistLabelsHeader labelsStream

            if imagesHeader.ImagesCount <> labelsCount then raise (Exception("Images and labals count must be equal."))

            let imageDataStartPosition = imagesStream.Position
            let labelsDataStartPosition = labelsStream.Position

            let dataSeq = seq {
                imagesStream.Position <- imageDataStartPosition
                let imagesSeq = mnistImagesSeq imagesHeader imagesStream
                
                labelsStream.Position <- labelsDataStartPosition
                let labelsSeq = mnistLabelsSeq labelsCount labelsStream 

                yield! Seq.zip labelsSeq imagesSeq
            }

            let mnistDisposableComposit = disposableComposite [imagesStream :> IDisposable; imageStorage; labelsStream :> IDisposable; labelsStorage] 
            imagesHeader, dataSeq, mnistDisposableComposit
        with | _ ->
            labelsStream.Dispose()
            labelsStorage.Dispose()
            reraise()
    with | _ ->
        imagesStream.Dispose()
        imageStorage.Dispose()
        reraise()    