<h1>Data Dictionary</h1>

<p>Yelp Dataset- Stars Prediction<br />
<strong>Aprendizaje Autom&aacute;tico</strong><br />
<strong>Master Universitario en Ciencia de Datos</strong></p>

<p>&nbsp;</p>

<p style="text-align:right">V&iacute;ctor Viloria V&aacute;zquez (<em>victor.viloria@cunef.edu</em>)</p>

<hr style="border:1px solid gray">

Este documento pretende aportar información descriptiva del contenido y forma del dataset utilizado en la práctica final de la asignatura de Machine Learning.

Este dataset incorpora informacion relativa de **seis categorías diferentes**:

**- Negocios (business)**

**- Valoraciones (review)**

**- Usuarios (user)**

**- Visitas (checkin)**

**- Reseñas (tip)**

**- Fotos (photo)**


**Para nuestro objetivo de negocio, unicamente utilizaremos los ficheros de `business` y `checkin`, por lo que describiremos ambos ficheros**

**Nos disponemos a describir cada una de sus variables, su significado, posibles valores que puede tomar y otros aspectos técnicos relevantes para el manejo y que permitan la elaboración de modelos.**.

<hr style="border:1px solid gray">

## Negocios (business)

Dentro de este dataset encontramos diferentes variables que facilitan información sobre los distintos locales/negocios registrados en la plataforma de Yelp

| **Field Name** 	| **Description** 	| **Variable Type** 	| **Data Type** 	| **Values** 	| **Number of Distinct values**|
|:---:	|:---	|---	|---	|:---	|:---:	|
| business_id 	| ID of business that belongs to the Yelp platform.<br>Composed of 22 unique characters 	| Categorical 	| _object_ 	| - Example: "mpf3x-BjTdTEA3yCZrAYPw" 	|34516|
| name 	| Business name 	| Categorical 	| _object_ 	|  - Example: "The UPS Store" 	|23102|
| adress 	| Full address of the business 	| Categorical 	| _object_ 	| - Example: "87 Grasso Plaza Shopping Center" 	|32340|
| city 	| City where the business is located 	| Categorical 	| _object_ 	| - Example: "Affton" 	|839|
| state 	| State code where the business is located.<br>(Composed of 2 characters, if applicable) 	| Categorical 	| _object_ 	| - Example: "MO" 	|16|
| postal_code 	| Postal code of the business 	| Categorical 	| _object_ 	| - Example: "63123" 	|3362|
| latitude 	| Latitude coordinates of the business 	| Categorical 	| _float64_ 	| - Example: 38.551126 	|135593|
| longitude 	| Longitude coordinates of the business 	| Categorical 	| _float64_ 	| - Example: -90.335695 	|131918|
| stars 	| Star rating, rounded to half-stars 	| Numerical 	| _float64_ 	| - Example: 3.0 	|9|
| review_count 	| Number of reviews received 	| Numerical 	| _int64_ 	| - Example: 15 	|1158|
| is_open 	| Clarifies if the business is still open or closed 	| Categorical 	| _int64_ 	| • 0: Closed<br>• 1: Open 	|2|
| attributes 	| Dictionary of business atributtes 	| Categorical 	| _object_ 	| • attributes_Business <br>• attributes_AcceptsCreditCards<br>  • attributes_BikeParking<br> • attributes_RestaurantsPriceRange2<br> • attributes_CoatCheck<br> • attributes_RestaurantsTakeOut<br>• attributes_RestaurantsDelivery<br>• attributes_Caters<br> • attributes_WiFi<br> • attributes_BusinessParking<br>• attributes_WheelchairAccessible<br> • attributes_HappyHour<br> • attributes_OutdoorSeating<br> • attributes_HasTV<br> • attributes_RestaurantsReservations<br> • attributes_DogsAllowed<br>• attributes_Alcohol<br> • attributes_GoodForKids<br>• attributes_RestaurantsAttire<br> • attributes_Ambience<br>• attributes_RestaurantsTableService<br> • attributes_RestaurantsGoodForGroups<br> • attributes_DriveThru<br>• attributes_NoiseLevel<br>• attributes_GoodForMeal<br>• attributes_BusinessAcceptsBitcoin<br>• attributes_Smoking<br>• attributes_Music<br>• attributes_GoodForDancing<br>• attributes_AcceptsInsurance<br>• attributes_BestNights<br>• attributes_BYOB<br>• attributes_Corkage<br>• attributes_BYOBCorkage<br>• attributes_HairSpecializesIn<br>• attributes_Open24Hours<br>• attributes_RestaurantsCounterService<br>• attributes_AgesAllowed<br>• attributes_DietaryRestrictions	|39|
| categories 	| Array of strings of business categories 	| Categorical 	| _object_ 	| - Example: 'Shipping Centers, Local Services, Notaries'  	|1311|
| hours 	| Weekly timeline of the business (24h) 	| Categorical 	| _object_ 	|  •	hours_Monday <br>•	hours_Tuesday<br>•	hours_Wednesday <br>• hours_Thursday<br>• hours_Friday<br>• hours_Saturday<br>• hours_Sunday<br>|7|




## Visitas (checkin)

Datos recopilados con las visitas publicadas en la plataforma de cada negocio.

| **Field Name** 	| **Description** 	| **Variable Type** 	| **Data Type** 	| **Values** 	| **Number of Distinct values**|
|:---:	|:---	|---	|---	|:---	|:---:	|
| business_id 	| ID of the business.<br>Composed of 22 unique characters 	| Categorical<br>(Int) 	| _object_ 	|- Example: ---kPU91CF4Lq2-WlRu9Lw  	|131930|
| date 	| Comma-separated list of timestamps for each checkin.<br>Format YYYY-MM-DD HH:MM:SS 	| Categorical<br>(String) 	|_object_  	|- Example: 2013-06-14 23:29:17, 2014-08-13 23:20:22  	|131930|
