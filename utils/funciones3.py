from geopy.distance import geodesic

# Sitios turísticos en New York City con sus coordenadas (latitud, longitud)
sitios_turisticos = {
    'statue_of_liberty': (40.6892, -74.0445),
    'central_park': (40.785091, -73.968285),
    'empire_state': (40.748817, -73.985428),
    'museo_metropolitano_de_arte_(met)': (40.7794, -73.9632),
    'time_square': (40.7580, -73.9855),
    'brooklyn_bridge': (40.7061, -73.9969),
    'vessel': (40.7532, -74.0020),
    'september_11_memorial': (40.7115, -74.0134),
    'madison_square_garden': (40.7505, -73.9934),
    'rockefeller_center': (40.7587, -73.9787),
    'museo_americano_historia_natural': (40.7813, -73.9735),
}

def distancia_minima_turistica(lat, lon):
    """Calcula la distancia mínima (en km) desde una coordenada a los sitios turísticos de NYC."""
    distancias = [
        geodesic((lat, lon), coords).km
        for coords in sitios_turisticos.values()
    ]
    return min(distancias)