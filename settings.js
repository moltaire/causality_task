// Experiment settings
const fixationDuration = 500; // this is in ms
const withinBlockBreakDuration = 30 * 1000; // in ms, too
const getReadyDuration = 5 * 1000;
const countdownDuration = 3 * 1000;
const breakAfter = 41; // how many trials until a break is shown
const choiceTimeout = 3000;
const feedbackColor = "Gold";
const stimBackgroundColor = "#999";
const stimForegroundColor = "#DDD";
const stimFrameColor = "#DDD";
var timeline = [];
var trialCounter = 1;
var repeatPractice = false;

// Eye tracking parameters
calibrationPointSize = 30;
calibrationPointDuration = 3000;
calibrationTimeToSaccade = 1000;
validationPointDuration = 3000;

// Quiz parameters
var nQuestions = 3; // How many questions from the pool to ask
var nCorrectRequired = 3; // How many questions have to be answered correctly
var nCorrect = 0;
var reminder_pages = [];
var questionSample = [];
// Draw random questions from the question pool
questionSample = jsPsych.randomization.sampleWithoutReplacement(
  questions,
  nQuestions
);
