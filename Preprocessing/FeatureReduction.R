#The code below was used to reduce the total number of features needed to perform the analysis
#The original dataset can be downloaded here: https://www.census.gov/programs-surveys/ahs/data/2017/ahs-2017-public-use-file--puf-/ahs-2017-national-public-use-file--puf-.html
AHS2017 <- read_csv("ahs2017n.csv")
AHSCO <- data.frame(AHS2017$MONOXIDE, AHS2017$CONTROL, AHS2017$HEATTYPE, AHS2017$HEATFUEL, AHS2017$HOTWATER, AHS2017$ACPRIMARY, AHS2017$ACSECNDRY, AHS2017$HINCP, AHS2017$HOA, AHS2017$YRBUILT, AHS2017$RENT, AHS2017$RENTCNTRL, AHS2017$HHMAR, AHS2017$HHGRAD)
AHSCO
names(AHSCO) <- c("MONOXIDE", "CONTROL", "HEATTYPE", "HEATFUEL", "HOTWATER", "ACPRIMARY", "ACSECNDRY", "HINCP", "HOA", "YRBUILT", "RENT", "RENTCNTRL", "HHMAR", "HHGRAD")
write_csv(AHSCO, "~/Documents/Vis/Advocacy/AHSCO.csv")
