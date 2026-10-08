# Brooklyn Bus Equity Explorer

An interactive tool for exploring Brooklyn bus service through an equity lens. Choose a date, route, direction or stop to see scheduled service, ridership and typical wait times on the map, alongside neighborhood need indicators and ranked lists of priority tracts, stops and routes. It follows the approach of equity-focused service reviews such as Title VI analysis, as an exploratory view of who is reached by service and who is not.

## Run locally

    pip install -r requirements.txt
    streamlit run app1.py

## Data and attribution

- Bus schedules: GTFS feed from the [Mobility Database](https://mobilitydatabase.org/feeds/gtfs/mdb-520), originally published by the MTA.
- Bus ridership: public data published by the MTA.
- Neighborhood demographics: U.S. Census Bureau, American Community Survey 5-year estimates.
- Basemap: map data © OpenStreetMap contributors (ODbL).

*This is my personal analysis of publicly available data, built to explore how transit service lines up with neighborhood need.*
