import UIKit
import Vision
import Accelerate
import Accelerate.vecLib.LinearAlgebra



public func getFaceLandmark(face_image: UIImage) -> VNFaceLandmarks2D? {
    
    let face_ciimage = CIImage(image: face_image)!
    
    let faceLandmarks = VNDetectFaceLandmarksRequest()
    try? VNSequenceRequestHandler().perform([faceLandmarks], on: face_ciimage)
    
    guard let landmarksResults = faceLandmarks.results as? [VNFaceObservation] else { return nil }
    
    
    return landmarksResults[0].landmarks
    
}




public func getFaceLandmarkImage(image_size: CGSize, faceLandmark: VNFaceLandmarks2D) -> UIImage {
    
    let rendererFormat = UIGraphicsImageRendererFormat()
    rendererFormat.scale = 1.0
    
    
    let image_rendered = UIGraphicsImageRenderer(size: image_size, format: rendererFormat).image { ctx in
        
        ctx.cgContext.setFillColor(gray: 0.0, alpha: 1.0)
        ctx.cgContext.fill(CGRect(origin: CGPoint.zero, size: image_size))
        ctx.cgContext.setFillColor(UIColor.cyan.cgColor)
        
        if let allpoints = faceLandmark.allPoints?.pointsInImage(imageSize: image_size) {
            
            for point in allpoints {
                let x = point.x-3
                let y = image_size.height - (point.y-3)
                
                ctx.cgContext.fillEllipse(in: CGRect(x: x, y: y, width: 6, height: 6))
            }
            
        }
        
    }
    
    return image_rendered
    
}



public func getAlignedFace(face_image: UIImage, faceLandmark: VNFaceLandmarks2D) -> UIImage? {
    
    let image_size = face_image.size
    let face_ciimage = CIImage(image: face_image)!
    
    
    guard let leftEye = faceLandmark.leftEye?.pointsInImage(imageSize: image_size),
          let rightEye = faceLandmark.rightEye?.pointsInImage(imageSize: image_size),
          let nose = faceLandmark.nose?.pointsInImage(imageSize: image_size) else { return nil }
    
    
    func invert(matrix : [Double]) -> [Double] {
        
        var inMatrix = matrix
        var N = __CLPK_integer(sqrt(Double(matrix.count)))
        var pivots = [__CLPK_integer](repeating: 0, count: Int(N))
        var workspace = [Double](repeating: 0.0, count: Int(N))
        var error : __CLPK_integer = 0
        
        withUnsafeMutablePointer(to: &N) {
            dgetrf_($0, $0, &inMatrix, $0, &pivots, &error)
            dgetri_($0, &inMatrix, $0, &pivots, &workspace, $0, &error)
        }
        return inMatrix
    }
    
    
    
    let x1 = Double(leftEye[0].x)
    let y1 = Double(image_size.height - leftEye[0].y)
    let x2 = Double(rightEye[4].x)
    let y2 = Double(image_size.height - rightEye[4].y)
    let x3 = Double(nose[4].x)
    let y3 = Double(image_size.height - nose[4].y)
    
    
    let vector = [18.639072, 16.249624,
                  75.73048, 15.18443,
                  47.515285, 49.38637]
    
    let matrix = [[x1, y1,  1,  0,  0,  0],
                  [ 0,  0,  0, x1, y1,  1],
                  [x2, y2,  1,  0,  0,  0],
                  [ 0,  0,  0, x2, y2,  1],
                  [x3, y3,  1,  0,  0,  0],
                  [ 0,  0,  0, x3, y3,  1]]
    
    
    let inv_matrix = invert(matrix: Array(matrix.joined()))
    var answer_matrix : [Double] = [0,0,0,0,0,0]
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                6, 1, 6,
                1.0, inv_matrix, 6,
                vector, 1,
                0.0, &answer_matrix, 1)
    
    print(answer_matrix)
    
    
    var transform = CGAffineTransform()
    transform.a = CGFloat(answer_matrix[0])
    transform.b = CGFloat(answer_matrix[3])
    transform.c = CGFloat(answer_matrix[1])
    transform.d = CGFloat(answer_matrix[4])
    transform.tx = CGFloat(answer_matrix[2])
    transform.ty = CGFloat(answer_matrix[5])
    
    
    
    let upsidedown_face = face_ciimage.transformed(by: CGAffineTransform(scaleX: 1, y: -1).concatenating(CGAffineTransform(translationX: 0, y: image_size.height)))
    let aligned_face = upsidedown_face.transformed(by: transform).cropped(to: CGRect(x: 0, y: 0, width: 96, height: 96))
    let result_face = aligned_face.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
    
    
    let context = CIContext.init(options: nil)
    return UIImage.init(cgImage: context.createCGImage(result_face, from: result_face.extent)!)
}
