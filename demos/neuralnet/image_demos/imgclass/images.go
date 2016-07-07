package main

import (
	"fmt"
	"image"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/unixpickle/num-analysis/linalg"
)

const (
	ImageDepth = 3
)

func LoadTrainingImages(dir string) (imgs map[string][]linalg.Vector, width, height int,
	err error) {
	dirContents, err := ioutil.ReadDir(dir)
	if err != nil {
		return
	}
	imgs = map[string][]linalg.Vector{}
	for _, item := range dirContents {
		if !item.IsDir() {
			continue
		}
		category := item.Name()
		subPath := filepath.Join(dir, item.Name())
		subContents, err := ioutil.ReadDir(subPath)
		if err != nil {
			return nil, 0, 0, err
		}
		for _, subItem := range subContents {
			if strings.HasPrefix(subItem.Name(), ".") {
				continue
			}
			imgPath := filepath.Join(subPath, subItem.Name())
			img, w, h, err := ReadImageFile(imgPath)
			if err != nil {
				return nil, 0, 0, fmt.Errorf("failed to read image %s: %s", imgPath,
					err.Error())
			} else if width == 0 && height == 0 {
				width = w
				height = h
			} else if w != width || h != height {
				return nil, 0, 0, fmt.Errorf("expected dimensions %dx%d but got %dx%d: %s",
					width, height, w, h, imgPath)
			}
			imgs[category] = append(imgs[category], img)
		}
	}
	return
}

func ReadImageFile(path string) (data linalg.Vector, width, height int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return
	}
	data = make(linalg.Vector, 0, img.Bounds().Dx()*img.Bounds().Dy()*ImageDepth)
	width = img.Bounds().Dx()
	height = img.Bounds().Dy()
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			data = append(data, float64(r)/0xffff, float64(g)/0xffff, float64(b)/0xffff)
		}
	}
	return
}
