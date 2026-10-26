// plate-crops harvests PlateCRNN training crops from labeled scene images.
//
// Usage:
//
//	plate-crops -config configs/config.yaml -in labels.csv -out crops/
//
// labels.csv rows are "plate,image_path". For every image, all detected plate
// regions are recognized; the region whose reading is closest to the label
// (edit distance <= -max-edit) is saved as <out>/<n>.png and listed in
// <out>/crops.csv as "label,png_path,pred,conf,edit". Crops are produced by
// the same detection + warp path the server uses, so training sees exactly
// what inference sees.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/vesaa/platex/internal/config"
	"github.com/vesaa/platex/internal/engine"
)

func main() {
	cfgPath := flag.String("config", "configs/config.yaml", "server config (model paths)")
	in := flag.String("in", "", "CSV of plate,image_path")
	out := flag.String("out", "", "output directory")
	maxEdit := flag.Int("max-edit", 2, "max edit distance between label and reading to accept a crop")
	workers := flag.Int("workers", 8, "parallel images")
	flag.Parse()
	if *in == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatal(err)
	}
	cfg.Engine.Workers = 1
	eng, err := engine.New(&cfg.Engine)
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	f, err := os.Open(*in)
	if err != nil {
		log.Fatal(err)
	}
	rows, err := csv.NewReader(f).ReadAll()
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	type result struct {
		row  []string
		crop engine.PlateCrop
		dist int
		ok   bool
	}
	results := make([]result, len(rows))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = result{row: rows[i]}
				img, err := loadImage(rows[i][1])
				if err != nil {
					log.Printf("skip %s: %v", rows[i][1], err)
					continue
				}
				crops, err := eng.DetectPlateCrops(img)
				if err != nil {
					log.Printf("skip %s: %v", rows[i][1], err)
					continue
				}
				best := -1
				for j, c := range crops {
					if d := editDistance(c.Plate, rows[i][0]); d <= *maxEdit && (best < 0 || d < results[i].dist) {
						best, results[i].dist = j, d
					}
				}
				if best >= 0 {
					results[i].crop, results[i].ok = crops[best], true
				}
			}
		}()
	}
	for i := range rows {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	lf, err := os.Create(filepath.Join(*out, "crops.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer lf.Close()
	cw := csv.NewWriter(lf)
	defer cw.Flush()
	n := 0
	for i, r := range results {
		if !r.ok {
			continue
		}
		p, _ := filepath.Abs(filepath.Join(*out, fmt.Sprintf("%06d.png", i)))
		if err := savePNG(p, r.crop.Image); err != nil {
			log.Fatal(err)
		}
		cw.Write([]string{r.row[0], p, r.crop.Plate, strconv.FormatFloat(float64(r.crop.Confidence), 'f', 4, 32), strconv.Itoa(r.dist)})
		n++
	}
	log.Printf("saved %d crops for %d images", n, len(rows))
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func editDistance(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	d := make([]int, len(rb)+1)
	for j := range d {
		d[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		prev := d[0]
		d[0] = i
		for j := 1; j <= len(rb); j++ {
			cur := d[j]
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			d[j] = min(d[j]+1, d[j-1]+1, prev+cost)
			prev = cur
		}
	}
	return d[len(rb)]
}
