import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.utils.proxy_call import OpenaiCall
from model.itinera_en import ItiNera
import argparse

app = Flask(__name__)
CORS(app)

# logging.basicConfig(level = logging.INFO)

# 设置日志级别为WARNING，减少信息量
logging.basicConfig(level=logging.WARNING)


def arg_parse(parser):
    parser.add_argument('--city', type = str, default ='shanghai',
                        choices = ["hangzhou", "qingdao", "shenzhen", "shanghai", "beijing", "changsha", "wuhan"],
                        help = 'Dataset city')
    return parser.parse_args()


@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    try:
        data = request.get_json()
        logging.info('Received data: %s', data)

        itinerary_request = data.get('request')
        if not itinerary_request:
            logging.warning('Empty itinerary request received')
            return jsonify({"error": "Itinerary request cannot be empty."}), 400

        city = data.get('city', 'shanghai')

        # 数据验证
        if not isinstance(itinerary_request, str) or not itinerary_request.strip():
            logging.error('Invalid itinerary request format')
            return jsonify({"error": "Invalid itinerary request format."}), 400
        if city not in ["hangzhou", "qingdao", "shenzhen", "shanghai", "beijing", "changsha", "wuhan"]:
            logging.error('Invalid city value')
            return jsonify({"error": "Invalid city value."}), 400

        # 数据预处理
        itinerary_request = itinerary_request.strip()

        # 调用 ItiNera 生成行程和路径
        day_planner = ItiNera(user_reqs=[itinerary_request], proxy_call=OpenaiCall(), city=city, type='en')
        logging.info('Starting to solve itinerary with request: %s, city: %s', itinerary_request, city)
        try:
            itinerary_info, lookup_dict = day_planner.solve()
        except ValueError as ve:
            logging.error('ValueError in ItiNera solve method: %s', str(ve))
            return jsonify({"error": f"ValueError in generating itinerary: {str(ve)}"}), 500
        except Exception as e:
            logging.error('Unexpected error in ItiNera solve method: %s', str(e))
            return jsonify({"error": f"Unexpected error in generating itinerary: {str(e)}"}), 500
        logging.info('Itinerary solved')
        logging.info(f'Itinerary: \n{itinerary_info}')

        # 解析返回的行程信息
        result_itinerary = []
        spots = []
        itinerary_steps = itinerary_info.get('itinerary_steps', [])

        if isinstance(itinerary_info, dict):
            duration = itinerary_info.get('duration', '1 day')
            pois = itinerary_info.get('pois', {})

            # 按 itinerary 的顺序重新排列 spots
            itinerary_names = itinerary_info.get('itinerary', '').split('->')
            temp_spots = []
            if isinstance(pois, dict):
                for key in sorted(pois.keys()):
                    poi = pois[key]
                    if isinstance(poi, dict):
                        lat = float(poi.get('latitude', 0))
                        lng = float(poi.get('longitude', 0))
                        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                            logging.warning('Invalid latitude or longitude in poi: %s', poi)
                            continue
                        spot = {
                            'name': poi.get('name', ''),
                            'address': poi.get('address', ''),
                            'category': poi.get('category', ''),
                            'rating': poi.get('rating', ''),
                            'description': poi.get('description', ''),
                            'latitude': lat,
                            'longitude': lng
                        }
                        temp_spots.append(spot)

            # 按 itinerary_names 重新排序 spots
            for name in itinerary_names:
                for spot in temp_spots:
                    if spot['name'] == name and spot not in spots:
                        spots.append(spot)
                        result_itinerary.append(name)
                        break

        # 计算路线的中心点
        if spots:
            valid_spots = [spot for spot in spots if -90 <= spot['latitude'] <= 90 and -180 <= spot['longitude'] <= 180]
            if valid_spots:
                center_lat = sum(spot['latitude'] for spot in valid_spots) / len(valid_spots)
                center_lng = sum(spot['longitude'] for spot in valid_spots) / len(valid_spots)
            else:
                city_coordinates = {
                    'shanghai': {'lat': 31.2304, 'lng': 121.4737},
                    'beijing': {'lat': 39.9042, 'lng': 116.4074},
                    # 可以添加更多城市的默认坐标
                }
                city_coord = city_coordinates.get(city, city_coordinates['shanghai'])
                center_lat = city_coord['lat']
                center_lng = city_coord['lng']
        else:
            city_coordinates = {
                'shanghai': {'lat': 31.2304, 'lng': 121.4737},
                'beijing': {'lat': 39.9042, 'lng': 116.4074},
                # 可以添加更多城市的默认坐标
            }
            city_coord = city_coordinates.get(city, city_coordinates['shanghai'])
            center_lat = city_coord['lat']
            center_lng = city_coord['lng']

        # 返回结果
        result = {
            "itinerary": result_itinerary,
            "duration": duration,
            "spots": spots,
            "centerLat": center_lat,
            "centerLng": center_lng,
            "itinerary_steps": itinerary_steps
        }

        logging.info('Returning result: %s', result)
        return jsonify(result)

    except Exception as e:
        logging.error('Error occurred: %s', str(e))
        return jsonify({"error": f"An error occurred while generating itinerary: {str(e)}"}), 500


if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    app.run(debug = True)