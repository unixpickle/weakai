package main

import (
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"os"
)

// Image is a black and white bitmap of pixels.
type Image struct {
	Vector []float64
	Width  int
	Height int
}

// ReadImage loads an image from an image file and generates a vector which represents its black and
// white pixels.
func ReadImage(path string) (*Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	image, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	res := &Image{
		Vector: make([]float64, image.Bounds().Dx()*image.Bounds().Dy()),
		Width:  image.Bounds().Dx(),
		Height: image.Bounds().Dy(),
	}
	idx := 0
	for y := 0; y < image.Bounds().Dy(); y++ {
		for x := 0; x < image.Bounds().Dx(); x++ {
			r, g, b, _ := image.At(x, y).RGBA()
			black := float64(r+g+b) / float64(0xffff*3)
			res.Vector[idx] = black
			idx++
		}
	}
	return res, nil
}

// Crop crops this image to a specific sub-region, returning an image for that sub-region.
func (i *Image) Crop(rect image.Rectangle) *Image {
	if rect.Min.X < 0 || rect.Min.Y < 0 || rect.Max.X > i.Width || rect.Max.Y > i.Height ||
		rect.Dx() < 0 || rect.Dy() < 0 {
		panic("invalid crop rectangle")
	}
	res := &Image{
		Vector: make([]float64, rect.Dx()*rect.Dy()),
		Width:  rect.Dx(),
		Height: rect.Dy(),
	}
	idx := 0
	for y := 0; y < rect.Dy(); y++ {
		for x := 0; x < rect.Dx(); x++ {
			res.Vector[idx] = i.At(x+rect.Min.X, y+rect.Min.Y)
			idx++
		}
	}
	return res
}

// At returns the brightness value at a given point.
func (i *Image) At(x, y int) float64 {
	if x < 0 || x >= i.Width || y < 0 || y >= i.Height {
		panic("point out of bounds")
	}
	return i.Vector[x+y*i.Width]
}

// WriteFile saves the image as a PNG at a path.
func (i *Image) WriteFile(path string) error {
	output, err := os.Create(path)
	if err != nil {
		return err
	}
	defer output.Close()
	img := image.NewRGBA(image.Rect(0, 0, i.Width, i.Height))
	for y := 0; y < i.Height; y++ {
		for x := 0; x < i.Width; x++ {
			brightness := i.At(x, y)
			val := uint8(brightness * 0xff)
			img.Set(x, y, color.RGBA{val, val, val, 0xff})
		}
	}
	return png.Encode(output, img)
}
