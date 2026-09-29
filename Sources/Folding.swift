// Coarse-grained folding: fixed-length hydrophobic/polar chain.
var foldSeed: UInt32 = 813
let positions = UnsafeMutablePointer<Double>.allocate(capacity: 84)
let candidate = UnsafeMutablePointer<Double>.allocate(capacity: 84)
var foldEnergy = 0.0, foldBest = 0.0
var foldSteps: Int32 = 0, foldAccepted: Int32 = 0
func hydrophobic(_ i: Int) -> Bool { i % 5 != 1 && i % 5 != 4 }
func energy(_ p: UnsafeMutablePointer<Double>) -> Double {
    var e = 0.0
    for i in 0..<26 { for j in (i+2)..<28 {
        var d2 = 0.0
        for a in 0..<3 { let delta = p[i*3+a]-p[j*3+a]; d2 += delta*delta }
        let d = d2.squareRoot()
        if d < 0.7 { e += 180*(0.7-d)*(0.7-d) }
        if hydrophobic(i) && hydrophobic(j) {
            let z = (d-1.1)/0.55; e -= exponential(-z*z)
        }
    }}
    return e
}
@_cdecl("fold_init") public func foldInit() {
    foldSeed = 813; foldSteps = 0; foldAccepted = 0
    for i in 0..<28 {
        positions[i*3] = Double(i)*0.85
        positions[i*3+1] = sine(Double(i)*0.9)*0.3
        positions[i*3+2] = cosine(Double(i)*0.9)*0.3
    }
    foldEnergy = energy(positions); foldBest = foldEnergy
}
@_cdecl("fold_step") public func foldStep(_ count: Int32) {
    for _ in 0..<max(0, min(count, 5000)) {
        let pivot = 1 + Int(random(&foldSeed)*25)
        let axis = Int(random(&foldSeed)*3)
        let angle = (random(&foldSeed)-0.5)*1.8
        let c = cosine(angle), s = sine(angle), a = (axis+1)%3, b = (axis+2)%3
        candidate.update(from: positions, count: 84)
        for i in (pivot+1)..<28 {
            let x = candidate[i*3+a]-candidate[pivot*3+a]
            let y = candidate[i*3+b]-candidate[pivot*3+b]
            candidate[i*3+a] = candidate[pivot*3+a]+x*c-y*s
            candidate[i*3+b] = candidate[pivot*3+b]+x*s+y*c
        }
        let next = energy(candidate), temperature = 0.07+0.9*exponential(-Double(foldSteps)/5000)
        if next < foldEnergy || random(&foldSeed) < exponential((foldEnergy-next)/temperature) {
            positions.update(from: candidate, count: 84); foldEnergy = next; foldAccepted += 1
        }
        foldSteps += 1; foldBest = min(foldBest, foldEnergy)
    }
}
@_cdecl("fold_positions") public func foldPositions() -> UnsafeMutablePointer<Double> { positions }
@_cdecl("fold_energy") public func currentEnergy() -> Double { foldEnergy }
@_cdecl("fold_best") public func bestEnergy() -> Double { foldBest }
@_cdecl("fold_steps") public func foldingSteps() -> Int32 { foldSteps }
@_cdecl("fold_accepted") public func foldingAccepted() -> Int32 { foldAccepted }
