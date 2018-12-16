package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func validate(x float64, fmt string, args ...interface{}) float64 {
	if math.IsNaN(x) {
		log.Panicf("NaN: "+fmt, args...)
	}
	if math.IsInf(x, 1) {
		log.Printf("+Inf: "+fmt, args...)
		return math.MaxFloat32
	}
	if math.IsInf(x, -1) {
		log.Printf("-Inf: "+fmt, args...)
		return -math.MaxFloat32
	}
	return x
}

func synthesize(n int, a, b float64) (x, y []float64) {
	for i := 0; i < n; i++ {
		α := validate(rand.Float64()*10.0, "synthesize α")
		β := validate(a*α+b+rand.Float64()/5.0, "synthesize β from α=%f",
			α)
		x = append(x, α)
		y = append(y, β)
	}
	return x, y
}

func forward(x, y []float64, a, b float64) (e []float64, c float64) {
	for i := range x {
		α := validate(a*x[i]+b-y[i], "forward e = %f * %f + %f - %f", a,
			x[i], b, y[i])
		e = append(e, α)
		c += α * α
	}
	return e, c / float64(len(x))
}

func backward(x, y, e []float64) (da, db float64) {
	for i := range x {
		da = validate(da+validate(e[i]*x[i], "backward e*x = %f * %f",
			e[i], x[i]),
			"da += e[i]*x[i]")
		db = validate(db+e[i], "backward db += e[i]")
	}
	n := float64(len(x))
	return validate(da*2/n, "backward return da=%f", da), validate(db*2/n,
		"backward return db=%f", db)
}

func optimize(a, b, da, db, η float64) (float64, float64) {
	a = validate(a-validate(da*η, "optimize da*η = %f * %f", da, η),
		"optimize a")
	b = validate(b-validate(db*η, "optimize db*η = %f * %f", db, η),
		"optimize b")
	return a, b
}

const (
	ta = 2.0 // true model parameters
	tb = 1.0
	η  = 0.01 // learning rate
	T  = 1000 // number of iterations
	m  = 16   // minibatch size
)

type model struct {
	a, b, c float64
}

func trainer(up, down chan model, freeTrialSteps int, curve *plotter.XYs) {
	a := 0.2 * rand.Float64() // random start
	b := 0.2 * rand.Float64()
	s := math.Inf(1) // mse >= 0
	step := freeTrialSteps
	for {
		x, y := synthesize(m, ta, tb) // minibatch
		e, c := forward(x, y, a, b)
		if step < freeTrialSteps {
			da, db := backward(x, y, e)
			a, b = optimize(a, b, da, db, η)
			step++
		} else {
			if c < s {
				s = c
				if up != nil {
					up <- model{a, b, c}
				}
			} else {
				if down != nil {
					θ := <-down
					a, b = θ.a, θ.b
				}
			}
			step = 0
		}
		*curve = append(*curve, struct{ X, Y float64 }{a, b})
	}
}

func ps(up, down chan model, curve *plotter.XYs) {
	var a, b float64
	c := math.Inf(1)
	updates := 0
	timer := time.After(2 * time.Second)
	for updates < 500 {
		select {
		case θ := <-up: // some worker uploaded a candidate model.
			updates++
			if θ.c < c { // looks good according to the worker.
				// double check with bigger dev set.
				x, y := synthesize(m*100, ta, tb)
				_, θ.c = forward(x, y, θ.a, θ.b)
				if θ.c < c {
					a, b, c = θ.a, θ.b, θ.c
					*curve = append(*curve,
						struct{ X, Y float64 }{a, b})
				}
			}
		case down <- model{a, b, c}: // response if any worker downloads.
		case <-timer:
			fmt.Println("Job stops after a certain period of time.")
			return
		}
	}
	fmt.Println("Job stops after the PS did a certain number of updates.")
}

func main() {
	const w = 10 // number of workers
	up := make(chan model)
	down := make(chan model)
	curves := make([]plotter.XYs, w+1)
	for i := 0; i < w; i++ {
		go trainer(up, down, 100, &curves[i])
	}
	ps(up, down, &curves[w])

	p, _ := plot.New()
	p.Title.Text = "Swamp Traces"
	p.X.Label.Text = "a"
	p.Y.Label.Text = "b"
	p.Add(plotter.NewGrid())
	for i := 0; i < w; i++ {
		plotutil.AddLinePoints(p, fmt.Sprintf("worker %d", i), curves[i])
	}
	ll, lp, _ := plotter.NewLinePoints(curves[w])
	ll.Color = color.RGBA{G: 255, A: 255}
	lp.Color = color.RGBA{G: 255, A: 255}
	s, _ := plotter.NewScatter(curves[w][len(curves[w])-1:])
	s.GlyphStyle.Color = color.RGBA{G: 255, A: 255}
	s.GlyphStyle.Shape = new(draw.BoxGlyph)
	p.Add(ll, s)
	p.Legend.Add("parameter server", ll, lp)
	p.Legend.Add("final model estimate", s)
	p.Save(10*vg.Inch, 5*vg.Inch, "traces.png")
}
