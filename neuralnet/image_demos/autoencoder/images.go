package main

import (
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"

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

func ImageFromTensor(t *neuralnet.Tensor3) image.Image {
	res := image.NewRGBA(image.Rect(0, 0, t.Width, t.Height))

	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			r := math.Min(math.Max(t.Get(x, y, 0), 0), 1)
			g := math.Min(math.Max(t.Get(x, y, 1), 0), 1)
			b := math.Min(math.Max(t.Get(x, y, 2), 0), 1)
			color := color.RGBA{
				A: 0xff,
				R: uint8(r * 0xff),
				G: uint8(g * 0xff),
				B: uint8(b * 0xff),
			}
			res.Set(x, y, color)
		}
	}

	return res
}

func ReadImages(path string) (<-chan image.Image, error) {
	dir, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer dir.Close()
	names, err := dir.Readdirnames(0)
	if err != nil {
		return nil, err
	}

	resChan := make(chan image.Image)
	go func() {
		for _, name := range names {
			imagePath := filepath.Join(path, name)
			f, err := os.Open(imagePath)
			if err != nil {
				continue
			}
			image, _, _ := image.Decode(f)
			f.Close()
			if image != nil {
				resChan <- image
			}
		}
		close(resChan)
	}()
	return resChan, nil
}
