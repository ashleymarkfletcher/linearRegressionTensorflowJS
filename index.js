// store mouse click points
const x_vals = []
const y_vals = []

// slope
let m
// y - intercept
let b

const learningRate = 0.3

// optimzer works on ALL tf variables if no sencond parameter is put in
const optimizer = tf.train.sgd(learningRate)

function setup() {
  createCanvas(400, 400)
  background(0)

  // init m and b as random values between 0 and 1
  // they are tf vriables because they change over time
  m = tf.variable(tf.scalar(random(1)))
  b = tf.variable(tf.scalar(random(1)))
}

const loss = (predictions, labels) =>
  predictions
    .sub(labels)
    .square()
    .mean()

const predict = x => {
  const xs = tf.tensor1d(x)

  // formula for a line
  // y = mx + b
  const ys = xs.mul(m).add(b)

  return ys
}

function mousePressed() {
  // normalise the graph from pixels
  let x = map(mouseX, 0, width, 0, 1)
  let y = map(mouseY, 0, width, 1, 0)

  x_vals.push(x)
  y_vals.push(y)
}

function draw() {
  background(0)
  stroke(255)
  strokeWeight(8)

  // get rid of all unneaded tensors to stop memory leak
  tf.tidy(() => {
    // don't train if no points
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals)
      optimizer.minimize(() => loss(predict(x_vals), ys))
    }
  })

  // draw the points
  for (let i = 0; i < x_vals.length; i++) {
    // undo the normalising
    const px = map(x_vals[i], 0, 1, 0, width)
    const py = map(y_vals[i], 0, 1, height, 0)

    point(px, py)
  }

  // points at the left and right 0-1
  const lineX = [0, 1]

  // predict the Y for each x point
  const ys = tf.tidy(() => predict(lineX))
  // get the values in the prediction
  const lineY = ys.dataSync()

  // map from normalised back to pixels
  // first point
  const x1 = map(lineX[0], 0, 1, 0, width)
  const x2 = map(lineX[1], 0, 1, 0, width)

  // second point
  const y1 = map(lineY[0], 0, 1, height, 0)
  const y2 = map(lineY[1], 0, 1, height, 0)

  strokeWeight(2)
  line(x1, y1, x2, y2)

  // get rid of ys since it's not needed
  ys.dispose()
}
