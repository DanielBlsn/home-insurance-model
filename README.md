Home Insurance Model

datasource: https://www.kaggle.com/ycanario/home-insurance
The raw data file has been removed due to size and should be added under data as home_insurance.csv for the workflow to work correctly.

This analysis and machine learning model was built as part of a technical test in roughly 2-3 days.

Structure:

+--data  
| +--home_insurance.csv  
+--models_data  
+--modules  
| +--data_enrichment.py  
| +--extract_data.py  
| +--ml_models.py  
| +--xgboost_optimization.py  
+--reports  
+--main.py  
+--build_reports.py  
+--build_classifier.py  
+--build_results.py  
+--create_reports.py  
+--create_charts.py  


Instructions:  

Run main.py to take the data through the whole ml workflow and hyperparameter optimization.  
Run build_reports.py to create pandas profiling initial reports.  
Run create_charts.py to create the data analysis charts.  

ml_models.py: to load models and make actual predictions.  
xgboost_optimization.py: does cross validation over folds for hyperparameter search.  

models_data: folder contains saved models and results  
reports: folder contains pandas profiling and analysis charts  


Raw Data Features and Notes:  

QUOTE_DATE: Day where the quotation was made   
COVER_START: Beginning of the cover payment(includes policies with no data -> exclude)  
CLAIM3YEARS: 3 last years loss (binary Y/N)  
P1EMPSTATUS: Client's professional status (multiple unknown categories)  
P1PTEMP_STATUS: Client's part-time professional status  (mostly missing values)  
BUS_USE: Commercial use indicator (binary Y/N)  
CLERICAL: Administration office usage indicator (binary Y/N, mostly missing values)  
AD_BUILDINGS: Building coverage - Self damage (binary Y/N, inconsistent with SUMINSUREDBUILDINGS when Y/0 sum)  
RISKRATEDAREA_B: Geographical Classification of Risk - Building (0-100)  
SUMINSUREDBUILDINGS: Assured Sum - Building (default 0 or 1 000 000)  
NCDGRANTEDYEARS_B: Bonus Malus - Building (0-9, years passed from last claim, inconsistent with CLAIM3YEARS)  
AD_CONTENTS: Coverage of personal items - Self Damage (binary Y/N)  
RISKRATEDAREA_C: Geographical Classification of Risk - Personal Objects (0-100)  
SUMINSUREDCONTENTS: Assured Sum - Personal Items   
NCDGRANTEDYEARS_C: Malus Bonus - Personal Items (0-9, years passed from last claim, inconsistent with CLAIM3YEARS)  
CONTENTS_COVER: Coverage - Personal Objects indicator (binary Y/N)  
BUILDINGS_COVER: Cover - Building indicator (binary Y/N)  
SPECSUMINSURED: Assured Sum - Valuable Personal Property  
SPECITEMPREM: Premium - Personal valuable items   
UNSPECHRPPREM: Related to premium paid by employer?   
P1_DOB: Date of birth of the client (multiple categories)  
P1MARSTATUS: Marital status of the client   
P1POLICYREFUSED: Police Emission Denial Indicator (binary Y/N)  
P1_SEX: customer sex  
APPR_ALARM: Appropriate alarm (binary Y/N)  
APPR_LOCKS: Appropriate lock (binary Y/N)  
BEDROOMS: Number of bedrooms  
ROOF_CONSTRUCTION: Code of the type of construction of the roof  
WALL_CONSTRUCTION: Code of the type of wall construction  
FLOODING: House susceptible to floods (binary Y/N)  
LISTED: National Heritage Building  
MAXDAYSUNOCC: Number of days unoccupied (7 categories)  
NEIGH_WATCH: Vigils of proximity present (binary Y/N)  
OCC_STATUS: Occupancy status (7 categories - unknown code meaning)  
OWNERSHIP_TYPE: Type of membership (unknown)  
PAYING_GUESTS: Presence of paying guests (binary 0/1)  
PROP_TYPE: Type of property (uknown codes)  
SAFE_INSTALLED: Safe installs (binary Y/N)  
SECDISCREQ: Reduction of premium for security (binary Y/N)  
SUBSIDENCE: Subsidence indicator (relative downwards motion of the surface) (binary Y/N)  
YEARBUILT: Year of construction  
CAMPAIGN_DESC: Description of the marketing campaign (all missing)  
PAYMENT_METHOD: Method of payment (3 categories)  
PAYMENT_FREQUENCY: Frequency of payment (mostly missing or 1)  
LEGALADDONPRE_REN: Option "Legal Fees" included before 1st renewal (binary Y/N)  
LEGALADDONPOST_REN: Option "Legal Fees" included after 1st renewal (binary Y/N)  
HOMEEMADDONPREREN: "Emergencies" option included before 1st renewal (binary Y/N)  
HOMEEMADDONPOSTREN: Option "Emergencies" included after 1st renewal (binary Y/N)  
GARDENADDONPRE_REN: Option "Gardens" included before 1st renewal (binary Y/N)  
GARDENADDONPOST_REN: Option "Gardens" included after 1st renewal (binary Y/N)  
KEYCAREADDONPRE_REN: Option "Replacement of keys" included before 1st renewal (binary Y/N)  
KEYCAREADDONPOST_REN: Option "Replacement of keys" included after 1st renewal (binary Y/N)  
HP1ADDONPRE_REN: Option "HP1" included before 1st renewal (binary Y/N)  
HP1ADDONPOST_REN: Option "HP1" included after 1st renewal (binary Y/N)  
HP2ADDONPRE_REN: Option "HP2" included before 1st renewal (binary Y/N)  
HP2ADDONPOST_REN: Option "HP2" included afterrenewal (binary Y/N)  
HP3ADDONPRE_REN: Option "HP3" included before 1st renewal (binary Y/N)  
HP3ADDONPOST_REN: Option "HP3" included after renewal (binary Y/N)  
MTA_FLAG: Mid-Term Adjustment indicator (binary Y/N)  
MTA_FAP: Bonus up to date of Adjustment (where present it mostly matches gross prem)  
MTA_APRP: Adjustment of the premium for Mid-Term Adjustmen (where present mostly matches MTA_FAP, otherwise actual adjustment)  
MTA_DATE: Date of Mid-Term Adjustment  
LAST_ANN_PREM_GROSS: Premium - Total for the previous year   
POL_STATUS: Policy status  
Policy: Policy number  
