// typescript logistic regression package

// perform logistic regression and return a prediction function
function logistic (X, y, alpha,iterations=5000,W=[]) {
    
    /*
        X is a 2d array of inputs
        y is a 1d array of answers
        alpha is the learning rate
        iterations is how many times to run regression
        W is the initial parameters of the model
    */

    // normalize each value of X
    let X_norm  = norm(X)

}

function cost_log (x,y) {
    
}

function zeroes(arr) { 
    // makes an array of length equal to the given array,
    // but filled with only zeroes (or replace)
    return Array.from("_".repeat(arr.length)).map(() => 0)
}

// calculate the mean of a 2D matrix
// tested
function mean(arr) {
    let result_arr = Array.from(Array.from("".repeat(arr[0].length))).map(() => 0)
    for (let j = 0; j < arr[0].length; j++) { // for each input (column)
        // console.log(`j is ${j}`)
        for (let i = 0; i < arr.length; i++){ // for each example (row)
            // console.log(`i is ${i}`)
            result_arr[j] = (arr.reduce((sum,nextval) => sum + nextval[j], 0)/arr.length)
            // console.log(`result_arr is now ${result_arr}`)
            
        }
    }
    return result_arr
}

// normalize a 2D Matrix
// tested
function norm(arr) {
    // store means of each column
    let means = mean(arr)
    // get min and max values
    // by making a list of vectors for each column
    // and calling Math.max/Math.min
    let vectors = zeroes(arr[0])
        .map(i => zeroes(arr))
    // console.log(vectors)
    for (let j = 0; j < arr[0].length; j++) { // for each column
            // console.log(`j is ${j}, so we're accessing the list ${vectors[j]}`)
        for (let i = 0; i < arr.length; i++) { // for each row
            // console.log(`i is ${i}`)
            // console.log(`BEFORE vectors[${j}] is ${vectors[j]}`)
            vectors[j][i] = arr[i][j]
            // console.log(`AFTER vectors[${j}] is ${vectors[j]}`)
            // console.log("\nthis is vectors right now:")
            // console.log(vectors)
            
        }
    }
    
    // console.log(vectors)
    let maxes = vectors.map(lon => Math.max(...lon))
    let mins = vectors.map(lon => Math.min(...lon))
    console.log(maxes)
    console.log(mins)
    // console.log(arr)
    // console.log(zeroes([1,2,3]))
    let resultArrList = zeroes(arr).map(() => zeroes(arr[0]))
    // console.log(resultArrList)
    for (let j = 0; j < arr[0].length; j++) { // for each column
        for (let i = 0; i < arr.length; i++) { // for each row
            resultArrList[i][j] = (arr[i][j] - means[j])/(maxes[j] - mins[j])
        }
    }
    return resultArrList
}

norm([[1,2],
      [3,4],
      [5,6]])

// debug notes:
// if you assign each element of a list to the same list, mutating any element will mutate them all. Because you're changing the single thing that all of them are pointing to. You'd need to make a copy or repeat the operation to do this cleanly.      