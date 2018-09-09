#!/usr/bin/env node

const csv = require('csv')
const _ = require('highland')
const fs = require('fs')
const { zip } = require('lodash/fp')

const inputPath = process.argv[2]

if (!inputPath) {
  console.error('No input path provided');
  process.exit(1)
}

const inputFile = fs.readFileSync(inputPath, 'utf-8')
const parsedFile = JSON.parse(inputFile)

let { data, envelope } = parsedFile

let d = envelope.labels.map((label, idx) => ([label, ...data[idx]]))
let header = ['Labels', ...envelope.columns]

let csvwriter = csv.stringify()
let outputFile = fs.createWriteStream(inputPath + '.csv')

_([header, ...d])
  .pipe(csvwriter)
  .pipe(outputFile)
