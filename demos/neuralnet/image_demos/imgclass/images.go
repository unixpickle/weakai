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
	ImageSize  = 96
	ImageDepth = 3
)

func LoadTrainingImages(dir string) (map[string][]linalg.Vector, error) {
	dirContents, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	res := map[string][]linalg.Vector{}
	for _, item := range dirContents {
		if !item.IsDir() {
			continue
		}
		category := item.Name()
		subPath := filepath.Join(dir, item.Name())
		subContents, err := ioutil.ReadDir(subPath)
		if err != nil {
			return nil, err
		}
		for _, subItem := range subContents {
			if strings.HasPrefix(subItem.Name(), ".") {
				continue
			}
			imgPath := filepath.Join(subPath, subItem.Name())
			img, err := ReadImageFile(imgPath)
			if err != nil {
				return nil, fmt.Errorf("failed to read image %s: %s", imgPath, err.Error())
			}
			res[category] = append(res[category], img)
		}
	}
	return res, nil
}

func ReadImageFile(path string) (linalg.Vector, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	if img.Bounds().Dx() != ImageSize || img.Bounds().Dy() != ImageSize {
		return nil, fmt.Errorf("expected image size %dx%d but got %dx%d",
			ImageSize, ImageSize, img.Bounds().Dx(), img.Bounds().Dy())
	}
	res := make(linalg.Vector, 0, img.Bounds().Dx()*img.Bounds().Dy()*ImageDepth)
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			res = append(res, float64(r)/0xffff, float64(g)/0xffff, float64(b)/0xffff)
		}
	}
	return res, nil
}
