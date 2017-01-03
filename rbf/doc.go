// Package rbf implements Radial Basis Function networks.
//
// A network consists of four independent components:
// distance computation, scaling, exponentiation, and
// linear classification.
//
// A network can be pre-trained using random sampling and
// least squares.
// Doing so is extremely easy if you already have an
// sgd.SampleSet that produces neuralnet.VectorSamples:
//
//     import "github.com/unixpickle/sgd"
//
//     ...
//
//     var samples sgd.SampleSet
//     var inDims, centerCount int
//     // Set samples and dimensions here.
//     net := &rbf.Network{
//         DistLayer:  rbf.NewDistLayerSamples(inDims, centerCount, samples),
//         ScaleLayer: rbf.NewScaleLayerShared(0.05),
//         ExpLayer:   &rbf.ExpLayer{Normalize: true},
//     }
//     net.OutLayer = rbf.LeastSquares(n.Net, samples, 20)
//
// You can also use stochastic gradient descent to train a
// network, or to fine-tune it after pre-training:
//
//     import "github.com/unixpickle/sgd"
//
//     ...
//
//     gradienter := &neuralnet.BatchRGradienter{
//         Learner:  net,
//         CostFunc: neuralnet.MeanSquaredCost{},
//     }
//     adam := &sgd.Adam{Gradienter: gradienter}
//     sgd.SGDInteractive(adam, samples, 0.001, 50, func() bool {
//         log.Println("Next training epoch...")
//         return true
//     })
//
package rbf
