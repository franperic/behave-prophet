CREATE TABLE behave (
  jobid TIMESTAMP NOT NULL,
  testtype VARCHAR(255) NOT NULL,
  run INT NOT NULL,
  fcstinit DATE NOT NULL,
  evalstep INT NOT NULL,
  ds DATE NOT NULL,
  yoriginal FLOAT NOT NULL,
  y FLOAT NOT NULL,
  yhat FLOAT,
  yhatlower FLOAT,
  yhatupper FLOAT,
  trend FLOAT,
  trendlower FLOAT,
  trendupper FLOAT,
  additiveterms FLOAT,
  additivetermslower FLOAT,
  additivetermsupper FLOAT,
  multiplicativeterms FLOAT,
  multiplicativetermslower FLOAT,
  multiplicativetermsupper FLOAT,
  yearly FLOAT,
  yearlylower FLOAT,
  yearlyupper FLOAT,
  simdate DATE,
  sim INT,
  error FLOAT,
  mape FLOAT
);



