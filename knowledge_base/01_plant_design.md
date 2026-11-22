# Plant design and layout

- The plant is a water-cooled chiller plant that serves a 12-floor large office building; each floor has one air-handling unit (AHU) with a chilled-water cooling coil.
- The plant has three identical chillers. Each chiller has its own constant-speed primary chilled-water pump and its own constant-speed condenser water pump.
- Primary/secondary arrangement: the three primary chilled-water loops (CWL_PRI) join and feed a secondary loop (CWL_SEC) with two identical variable-speed pumps. A bypass (decoupler) line between the loops balances primary and secondary flow.
- Condenser side: the three condenser water loops join (CDWL) and feed three identical cooling towers (CT). A three-way mixing valve (TWV_CTRL) lets part of the condenser water bypass the towers.
- Equipment runs in sets: starting one chiller also starts one cooling tower, one condenser water pump and one primary chilled-water pump. Unit numbers _1, _2, _3 in variable names identify the set.
- Units in the data: temperatures in degF, water flows in GPM, power in kW, the secondary-loop cooling load (CWL_SEC_LOAD) in W, differential pressure in inH2O; control signals, speeds and valve position are fractions 0-1; status points are 0 = off, 1 = on.
- Unit conversions: a temperature difference of 1 degC equals 1.8 degF; 1 ton of refrigeration = 3.517 kW = 12,000 Btu/h; cooling load in tons ~ GPM x delta T (degF) / 24.
- Chiller COP = cooling output (kW) / chiller electrical input (kW). Plant COP = cooling load / total plant power (chillers + cooling-tower fans + all pumps), so plant COP is always lower than chiller COP.
- Cooling-tower approach = tower leaving water temperature minus outdoor wet-bulb temperature. Cooling-tower range = tower entering minus leaving water temperature.
