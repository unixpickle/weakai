package main

import (
	"bytes"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/nfnt/resize"
)

const (
	// Download http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt
	// and remove the comments at the top of the file.
	FilePath = "dev_urls.txt"

	OutputDir    = "./faces"
	RoutineCount = 10
	OutputSize   = 96
)

type requestInfo struct {
	index int
	line  string
}

func main() {
	dbData, err := ioutil.ReadFile(FilePath)
	if err != nil {
		panic(err)
	}

	reqChan := make(chan requestInfo)
	var wg sync.WaitGroup

	for i := 0; i < RoutineCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for req := range reqChan {
				parts := strings.Split(req.line, "\t")
				url := parts[2]
				rectParts := strings.Split(parts[3], ",")
				var rectNums [4]int
				for i, r := range rectParts {
					n, _ := strconv.Atoi(r)
					rectNums[i] = n
				}
				face, err := fetchFace(url, rectNums)
				if err != nil {
					log.Printf("Error for face %d: %s", req.index, err.Error())
				} else {
					imageData := encodeImage(face)
					filePath := filepath.Join(OutputDir, strconv.Itoa(req.index)+".png")
					if err := ioutil.WriteFile(filePath, imageData, 0755); err != nil {
						log.Printf("Error writing: %s", filePath)
					}
				}
			}
		}()
	}

	lines := strings.Split(string(dbData), "\n")
	for i, line := range lines {
		reqChan <- requestInfo{i, line}
	}

	close(reqChan)
	wg.Wait()
}

func fetchFace(url string, rect [4]int) (image.Image, error) {
	log.Println("fetching", url)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	img, _, err := image.Decode(resp.Body)
	if err != nil {
		return nil, err
	}

	cropped := croppedImage{img, rect}
	return resize.Resize(OutputSize, OutputSize, cropped, resize.Bilinear), nil
}

func encodeImage(img image.Image) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic("failed to encode PNG: " + err.Error())
	}
	return buf.Bytes()
}

type croppedImage struct {
	img    image.Image
	bounds [4]int
}

func (c croppedImage) ColorModel() color.Model {
	return c.img.ColorModel()
}

func (c croppedImage) At(x, y int) color.Color {
	return c.img.At(x, y)
}

func (c croppedImage) Bounds() image.Rectangle {
	b1 := c.img.Bounds()
	b1.Min.X = c.bounds[0]
	b1.Min.Y = c.bounds[1]
	b1.Max.X = c.bounds[2]
	b1.Max.Y = c.bounds[3]
	return b1
}
