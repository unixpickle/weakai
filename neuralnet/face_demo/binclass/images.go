package main

import (
	"image"

	"github.com/unixpickle/weakai/neuralnet"
)

func ImageTensor(img image.Image) *neuralnet.Tensor3 {
	res := neuralnet.NewTensor3(img.Bounds().Dx(), img.Bounds().Dy(), 3)
	for y := 0; y < res.Height; y++ {
		for x := 0; x < res.Width; x++ {
			r, g, b, _ := img.At(x+img.Bounds().Min.X, y+img.Bounds().Min.Y).RGBA()
			res.Set(x, y, 0, float64(r)/65535.0)
			res.Set(x, y, 1, float64(g)/65535.0)
			res.Set(x, y, 2, float64(b)/65535.0)
		}
	}
	return res
}
