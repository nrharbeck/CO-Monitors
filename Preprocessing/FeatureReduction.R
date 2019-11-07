#The code below was used to reduce the total number of features needed to perform the analysis
#The original dataset can be downloaded here: https://www.census.gov/programs-surveys/ahs/data/2017/ahs-2017-public-use-file--puf-/ahs-2017-national-public-use-file--puf-.html
library(readr)
AHS2017 <- read_csv("ahs2017n.csv")
AHSCO <- data.frame(AHS2017$MONOXIDE, AHS2017$CONTROL, AHS2017$HEATTYPE, AHS2017$HEATFUEL, AHS2017$HOTWATER, AHS2017$ACPRIMARY, AHS2017$ACSECNDRY, AHS2017$HINCP, AHS2017$HOA, AHS2017$YRBUILT, AHS2017$RENT, AHS2017$RENTCNTRL, AHS2017$HHMAR, AHS2017$HHGRAD, AHS2017$HHRACE, AHS2017$OMB13CBSA, AHS2017$INTLANG, AHS2017$WEIGHT)
AHSCO
names(AHSCO) <- c("MONOXIDE", "CONTROL", "HEATTYPE", "HEATFUEL", "HOTWATER", "ACPRIMARY", "ACSECNDRY", "HINCP", "HOA", "YRBUILT", "RENT", "RENTCNTRL", "HHMAR", "HHGRAD", "HHRACE", "OMB13CBSA", "INTLANG", "WEIGHT")
write_csv(AHSCO, "~/Documents/Vis/Advocacy/AHSCO.csv")
AHSCO

#The below file is a subseted version of the full data dictionary from the Census Bureau. Columns matching the list from above were isolated to make the file more manageable.
ValueLabs <- read_csv("ValueLabels.csv")
Labels <- subset(ValueLabs, NAME == "MONOXIDE" | NAME == "CONTROL" | NAME == "HEATTYPE" | NAME == "HEATFUEL" | NAME == "HOTWATER" | NAME == "ACPRIMARY" | NAME == "ACSECNDRY" | NAME == "HINCP" | NAME == "HOA" | NAME == "YRBUILT" | NAME == "RENT" | NAME == "RENTCNTRL" | NAME == "HHMAR" | NAME == "HHGRAD" | NAME == "HHRACE" | NAME == "OMB12CBSA" | NAME == "INTLANG" | NAME == "WEIGHT")
Labels
write_csv(Labels, "~/Documents/Vis/Advocacy/ValueLabels.csv")
