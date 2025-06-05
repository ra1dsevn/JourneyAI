import os
import re
import logging
import copy
import json
import time
import scipy
import folium
import datetime
import numpy as np
import concurrent.futures
from xyconvert import gcj2wgs

from model.utils.funcs import RecurringList, compute_consecutive_distances, find_indices, sample_items, reorder_list, \
    remove_duplicates
from model.utils.all_en_prompts import get_start_point_prompt, get_dayplan_prompt, get_system_prompt, get_hour_prompt, \
    check_final_reverse_prompt, process_input_prompt
from model.utils.funcs import get_user_data_embedding  # this is only for the open-source version
from model.search import SearchEngine
from model.spatial import SpatialHandler


class ItiNera:
    def __init__(self, user_reqs, min_poi_candidate_num=19, keep_prob=0.8, thresh=1000, hours=0, proxy_call=None,
                 citywalk=True, city=None, type='zh'):

        # Initialize core parameters and constants
        self.MODEL = "gpt-4o-mini"
        # (hours, poi_num, distance_thresh)
        self.TIME2NUM = {1: (1, 3, 2000), 2: (1, 5, 3000), 3: (2, 7, 4000), 4: (2, 9, 5000), 5: (3, 11, 6000),
                         6: (3, 13, 7000), 7: (4, 15, 8000), 8: (4, 17, 9000)}

        self.min_poi_candidate_num = min_poi_candidate_num

        # Process user requirements and set parsed input data
        self.type = type
        self.proxy = proxy_call
        self.citywalk = citywalk
        self.user_reqs = user_reqs
        self.keep_prob = keep_prob
        self.thresh = thresh
        self.hours = hours if hours > 0 else 4  # 直接设置默认值 4，避免调用 get_hours

        # Process response data and initialize user data
        parsed_resquest = self.parse_user_request(user_reqs)
        self.must_see_poi_names, self.itinerary_pos_reqs, self.itinerary_neg_reqs, self.user_pos_reqs, self.user_neg_reqs, self.start_poi, self.end_poi = self.parse_user_input(
            parsed_resquest)

        # Initialize embeddings and user data
        self.user_favorites, self.embedding, self.r2i, self.i2r, self.must_see_pois = get_user_data_embedding(
            city_name=city, must_see_poi_names=self.must_see_poi_names, type=type)

        # Initialize spatial and search components based on hours
        self.maxPoiNum = self.TIME2NUM[self.hours][1]
        self.search_engine = SearchEngine(embedding=self.embedding, proxy=self.proxy)
        self.spatial_handler = SpatialHandler(data=self.user_favorites, min_clusters=self.TIME2NUM[self.hours][0],
                                              min_pois=self.maxPoiNum, citywalk=self.citywalk,
                                              citywalk_thresh=self.TIME2NUM[self.hours][2])

    def save_qualitative(self, new_numerical_order, full_response, clusters, lookup_dict, full_response_data):
        """
        Generates and saves interactive maps based on a new POI order and cluster information.
        The maps display markers, paths, and cluster centroids, saved to HTML files.

        Args:
            new_numerical_order (list): Ordered list of POI indices.
            full_response (str): JSON string containing POI response data.
            clusters (list): List of clusters with POI IDs.
            lookup_dict (dict): Dictionary for looking up POI names and IDs.
            full_response_data (dict): Dictionary containing full response data with coordinates.
        """

        # Helper to create and save map with POI markers and paths
        def create_map(df, polyline_color="green", map_filename=""):
            # 直接使用原始的GCJ - 02坐标
            coords_list = df[['lon', 'lat']].values.tolist()
            coords_json = json.dumps(coords_list)
            html_content = """
            <!doctype html>
            <html>
            <head>
                <meta charset="utf - 8">
                <meta http - equiv="X - UA - Compatible" content="IE=edge">
                <meta name="viewport" content="initial - scale = 1.0, user - scalable = no, width = device - width">
                <style>
                html,
                body,
                #container {
                    width: 100%;
                    height: 100%;
                }
                </style>
                <title>根据规划数据绘制步行路线</title>
                <script type="text/javascript">
                    window._AMapSecurityConfig = {
                      securityJsCode: "e0d9ecd2f840a7140dd370649592a6e8",
                    };
                  </script>
                <link rel="stylesheet" href="https://a.amap.com/jsapi_demos/static/demo - center/css/demo - center.css" /> 
                <script src="https://webapi.amap.com/maps?v = 1.4.15&key = 02f90d4b9c0cb23602e9a02786971af7&plugin = AMap.PolyEditor"></script>
                <script src="https://webapi.amap.com/maps?v = 1.4.10&key = 02f90d4b9c0cb23602e9a02786971af7&plugin = AMap.Walking"></script>
                <script src="https://a.amap.com/jsapi_demos/static/demo - center/js/demoutils.js"></script>
            </head>
            <body>
            <div id="container"></div>
            <script type="text/javascript">
                var map = new AMap.Map("container", {
                    center: [116.397559, 39.89621],
                    zoom: 14,
                    lang:"en"
                });

                // 当前示例的目标是展示如何根据规划结果绘制路线，因此walkOption为空对象
                var walkingOption = {}

                // 步行导航
                var walking = new AMap.Walking(walkingOption);
                """ + f"var data = {coords_json};" + """
                // 根据起终点坐标规划步行路线
                for (var i = 0; i < data.length - 1; i++) {
                    (function (index) { // 使用闭包确保正确的索引
                        walking.search(data[index], data[index + 1], function (status, result) {
                            if (status === 'complete') {
                                if (result.routes && result.routes.length) {
                                    drawRoute(result.routes[0], index); // 传递当前索引
                                    console.log('绘制步行路线完成');
                                } else {
                                    console.error('步行路线数据查询成功，但未找到路线: ', result);
                                }
                            } else {
                                console.error('步行路线数据查询失败: ', status, result);
                            }
                        });
                    })(i);
                }

                function drawRoute(route, index) {
                    var path = parseRouteToPath(route);

                    // 创建起点标记
                    if (index === 0) {
                        var startMarker = new AMap.Marker({
                            position: path[0],
                            icon: 'https://webapi.amap.com/theme/v1.3/markers/n/start.png',
                            map: map
                        });
                    }

                    // 创建途经点标记
                    if (index > 0 && index < data.length - 1) {
                        var viaMarker = new AMap.Marker({
                            position: path[index], // 途经点的实际位置
                            icon: 'https://a.amap.com/jsapi_demos/static/demo - center/icons/dir - via - marker.png',
                            map: map
                        });
                    }

                    // 创建终点标记
                    if (index === data.length - 2) {
                        var endMarker = new AMap.Marker({
                            position: path[path.length - 1],
                            icon: 'https://webapi.amap.com/theme/v1.3/markers/n/end.png',
                            map: map
                        });
                    }

                    // line
                    var routeLine = new AMap.Polyline({
                        path: path,
                        isOutline: true,
                        outlineColor: '#ffeeee',
                        borderWeight: 2,
                        strokeWeight: 5,
                        strokeColor: '#0091ff',
                        lineJoin: 'round'
                    });

                    routeLine.setMap(map);

                    // 调整视野达到最佳显示区域
                    var markers = [startMarker, endMarker, viaMarker].filter(marker => marker); // 过滤掉未定义的标记
                    map.setFitView(markers.concat(routeLine)); // 调整视野
                }

                // 解析WalkRoute对象，构造成AMap.Polyline的path参数需要的格式
                function parseRouteToPath(route) {
                    var path = [];

                    for (var i = 0; i < route.steps.length; i++) {
                        var step = route.steps[i];

                        for (var j = 0; j < step.path.length; j++) {
                            path.push(step.path[j]);
                        }
                    }

                    return path;
                }
            </script>
            </body>
            </html> """
            output_path = rf"./model/output/{map_filename}_Amap.html"
            with open(output_path, "w", encoding="utf - 8") as file:
                file.write(html_content)

            # 创建地图使用原始的GCJ - 02坐标
            m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)

            # Add markers to the map
            for i, (_, row) in enumerate(df.iterrows()):
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"{i} - {row['name']}",
                    icon=folium.Icon(color="blue", icon="info - sign")
                ).add_to(m)

            # Add polyline route between points
            points = list(zip(df['lat'], df['lon']))
            folium.PolyLine(points, color=polyline_color, weight=2.5, opacity=1).add_to(m)

            # Save the map to the specified file path
            m.save(f'./model/output/{map_filename}')

        # Generate current timestamp for filenames
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        current_time += f"_{self.type}"

        cluster_list, unique_clusters = [], set()

        # Extract clusters for each POI in the response
        for key in full_response_data["pois"].keys():
            poi_id, name = lookup_dict[int(key)]
            for cluster_idx, cluster in enumerate(clusters):
                if self.i2r[poi_id] in cluster:
                    cluster_list.append((cluster_idx, poi_id, name))
                    unique_clusters.add(cluster_idx)

        # Generate the initial full TSP map with all POIs in the new order
        # df_full_order = self.user_favorites.loc[new_numerical_order]
        # create_map(df_full_order, map_filename=f"{current_time}_fulltsp.html")

        # Generate map focused on POIs listed in the response
        response_indices = np.array(new_numerical_order)[
            np.array(list(full_response_data["pois"].keys())).astype(int) - 1]
        df_response_order = self.user_favorites.loc[response_indices]
        create_map(df_response_order, map_filename=f"{current_time}.html")

        # Obtain centroids and cluster radius settings
        centroids = self.spatial_handler.get_cluster_centroids(clusters, lonlat=True)
        radius = self.spatial_handler.citywalk_thresh if self.mark_citywalk else self.thresh

        # Add circles around cluster centroids to indicate cluster boundaries
        m = folium.Map(location=[df_response_order['lat'].mean(), df_response_order['lon'].mean()], zoom_start=13)
        for cluster_id in unique_clusters:
            centroid_lon, centroid_lat = centroids[cluster_id]
            folium.Circle(
                location=[centroid_lat, centroid_lon],
                radius=radius / 2,
                color='blue',
                fill=True
            ).add_to(m)

        # Overlay the polyline path for POIs in the response
        points = list(zip(df_response_order['lat'], df_response_order['lon']))
        folium.PolyLine(points, color="green", weight=2.5, opacity=1).add_to(m)
        m.save(f'./model/output/{current_time}_response_clusters.html')

        with open(f'./model/output/result_{self.type}.json', "w", encoding="utf - 8") as f:
            json.dump(full_response_data, f, ensure_ascii=True, indent=4)

        return full_response_data

    def get_hours(self, user_reqs, hours):
        """Get the number of hours for the plan; fetch from proxy if not provided."""
        if hours == 0:
            msg = [{"role": "user", "content": get_hour_prompt(user_reqs=user_reqs)}]
            try:
                response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
                # 尝试解析 JSON
                hours_list = json.loads(response)
                hours = int(hours_list[0])
            except json.JSONDecodeError:
                # 如果不是 JSON，尝试从文本中提取数字
                import re
                match = re.search(r'\["(\d+)"\]', response)  # 匹配 ["4"] 格式中的数字
                if match:
                    hours = int(match.group(1))
                else:
                    hours = 4  # 默认值
            # 确保 hours 在 1-8 范围内
            hours = max(1, min(8, hours))
        return hours

    def parse_user_request(self, user_reqs):
        """Fetch and parse user response from the proxy."""
        response = self.proxy.chat(messages=[{"role": "user", "content": process_input_prompt(user_input=user_reqs)}],
                                   model=self.MODEL).replace("'", '"')
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结构
            logging.warning(f"Failed to parse response as JSON: {response}. Using default structure.")
            match = re.search(r'from\s+(.+?)\s+to\s+(.+)', user_reqs[0], re.IGNORECASE)
            if match:
                start_poi = match.group(1).strip()
                end_poi = match.group(2).strip()
                return [
                    {"pos": start_poi, "neg": None, "mustsee": True, "type": "starting point"},
                    {"pos": end_poi, "neg": None, "mustsee": True, "type": "ending point"},
                    {"pos": "art and museum spots", "neg": None, "mustsee": False, "type": "location"}
                ]
            return []

    def parse_user_input(self, structured_input):
        must_see_poi_names = []
        itinerary_pos_reqs, itinerary_neg_reqs = [], []
        user_pos_reqs, user_neg_reqs = [], []
        start_poi, end_poi = None, None

        for req in structured_input:
            if req["type"] is None:
                req["type"] = "location"
            if req["type"] == "itinerary":
                itinerary_pos_reqs.append(req["pos"])
                if req["neg"] is not None:
                    itinerary_neg_reqs.append(req["neg"])
            elif req["type"] in ["location", "starting point", "ending point"]:
                if req["mustsee"] == True:
                    must_see_poi_names.append(req["pos"])
                user_pos_reqs.append(req["pos"])
                user_neg_reqs.append(req["neg"])
                if req["type"] == "starting point":
                    start_poi = req["pos"]
                if req["type"] == "ending point":
                    end_poi = req["pos"]

        # 如果未明确解析起点和终点，尝试从用户输入中提取“from A to B”模式
        if not start_poi or not end_poi:
            match = re.search(r'from\s+(.+?)\s+to\s+(.+)', self.user_reqs[0], re.IGNORECASE)
            if match:
                start_poi = match.group(1).strip()
                end_poi = match.group(2).strip()

        if len(user_pos_reqs) == 0:
            user_pos_reqs = itinerary_pos_reqs

        logging.info(f"Parsed start_poi: {start_poi}, end_poi: {end_poi}")
        self.start_poi = start_poi  # 存储到实例变量
        self.end_poi = end_poi  # 存储到实例变量
        return must_see_poi_names, itinerary_pos_reqs, itinerary_neg_reqs, user_pos_reqs, user_neg_reqs, start_poi, end_poi

    def get_reqs_topk(self):
        """
        Retrieves the top-k POIs for each user request and aggregates their scores.

        Returns:
            tuple: A sorted numpy array with unique POIs and their accumulated scores,
                and a list of pseudo-must-see POIs.
        """

        def process_request(user_pos_req, user_neg_req):
            # Limit top-k to the minimum of available POIs or the defined candidate number
            top_k = min(self.user_favorites.shape[0], self.min_poi_candidate_num)
            req_pois = self.search_engine.query(desc=(user_pos_req, user_neg_req), top_k=top_k)

            # Collect top two POIs as pseudo-must-see if not already present
            pseudo_must_see_local = [int(poi) for poi in req_pois[:2, 0] if poi not in pseudo_must_see_pois]
            return req_pois, pseudo_must_see_local

        all_reqs_topk, result, pseudo_must_see_pois = [], [], []

        if len(self.user_pos_reqs) > 1:
            # Use a thread pool for concurrent processing of multiple positive requests
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for req_pois, pseudo_must_see_local in executor.map(process_request, self.user_pos_reqs,
                                                                    self.user_neg_reqs):
                    pseudo_must_see_pois.extend(pseudo_must_see_local)
                    all_reqs_topk.append(req_pois)
        elif len(self.user_pos_reqs) == 1:
            # Handle single request case directly
            neg_req = self.user_neg_reqs[0] if self.user_neg_reqs else None
            req_pois, pseudo_must_see_local = process_request(self.user_pos_reqs[0], neg_req)
            pseudo_must_see_pois.extend(pseudo_must_see_local)
            all_reqs_topk.append(req_pois)
        else:
            raise ValueError("No positive requests found")

        # Concatenate results and aggregate scores for unique POIs
        all_reqs_topk = np.concatenate(all_reqs_topk, axis=0)
        unique_values = np.unique(all_reqs_topk[:, 0])
        result = [[value, all_reqs_topk[all_reqs_topk[:, 0] == value][:, 1].sum()] for value in unique_values]
        result = np.array(result)

        # Sort by score in descending order
        sorted_reqs_topk = result[result[:, 1].argsort()[::-1]]

        return sorted_reqs_topk, pseudo_must_see_pois

    def get_poi_candidates(self, req_topk_pois: np.ndarray, must_see_poi_idlist: list, pseudo_must_see_pois):
        """
        Selects POI candidates based on top-k requested POIs, must-see POIs, and pseudo-must-see POIs,
        while maintaining spatial clustering.

        Args:
            req_topk_pois (np.ndarray): Array of top-k POIs from user requests.
            must_see_poi_idlist (list): List of must-see POI IDs.
            pseudo_must_see_pois (list): List of pseudo-must-see POI IDs.

        Returns:
            tuple: A list of POI pairs across clusters, selected clusters, cluster order,
                POI candidates array, and POI candidate scores array.
        """

        req_topk_poi_idList = req_topk_pois[:, 0].astype(int).tolist()
        req_topk_poi_idList.extend(must_see_poi_idlist)

        all_poi_idlist = list(set(req_topk_poi_idList))
        poi_candidates, poi_candidate_scores, selected_cluster, self.mark_citywalk = self.spatial_handler.get_poi_candidates(
            all_poi_idlist, must_see_poi_idlist, req_topk_pois, self.min_poi_candidate_num, self.thresh,
            pseudo_must_see_pois)

        if min(1, self.min_poi_candidate_num / len(poi_candidates)) < 1:
            self.keep_prob = self.min_poi_candidate_num / len(poi_candidates)
            poi_candidates, poi_candidate_scores, selected_cluster = sample_items(poi_candidates, poi_candidate_scores,
                                                                                  selected_cluster,
                                                                                  keep_prob=self.keep_prob,
                                                                                  keep_ids=pseudo_must_see_pois)  # add some randomness
            selected_cluster = [sublist for sublist in selected_cluster if sublist]

        new_selected_cluster = copy.deepcopy(selected_cluster)
        for poi in poi_candidates:
            mark_included = False
            for cluster in selected_cluster:
                if poi in cluster:
                    mark_included = True
                    break
            if not mark_included:
                new_selected_cluster.append([poi])
        selected_cluster = new_selected_cluster

        order = reorder_list(poi_candidates, selected_cluster)
        poi_candidates, poi_candidate_scores = np.array(poi_candidates)[order], np.array(poi_candidate_scores)[order]
        clusterCentroids = self.spatial_handler.get_cluster_centroids(selected_cluster)
        clusters_order, _, _ = self.spatial_handler.get_tsp_order(locs=np.array(clusterCentroids))

        newclusters_order = []
        recurring_order = [i for i in clusters_order]
        recurring_order.append(recurring_order[0])
        distances = compute_consecutive_distances(np.array(clusterCentroids), recurring_order)
        topmax_distance_ids = distances.argsort()[-1:][::-1][0]  # k=1
        newclusters_order.extend(clusters_order[topmax_distance_ids + 1:])
        newclusters_order.extend(clusters_order[:topmax_distance_ids + 1])
        clusters_order = newclusters_order

        all_pairs = self.spatial_handler.get_poi_pairs_across_clusters(clusters_order,
                                                                       selected_cluster)  # those pairs are base on the reordered clusters

        return all_pairs, selected_cluster, clusters_order

    def calculate_ordered_route_info(self, new_numerical_order):
        distance_numerical_order = copy.deepcopy(new_numerical_order)
        distance_numerical_order.append(distance_numerical_order[0])
        points = self.user_favorites.loc[distance_numerical_order][["x", "y"]].astype("float").to_numpy()
        squared_diffs = np.diff(points, axis=0) ** 2
        distances = np.sqrt(np.sum(squared_diffs, axis=1))
        distance_numerical_order_poiname = self.user_favorites.loc[distance_numerical_order, "name"].tolist()

        distance_string = ""
        for i in range(len(distances)):
            distance_string += f"'{distance_numerical_order_poiname[i]}' is {int(distances[i])} meters away from '{distance_numerical_order_poiname[i + 1]}'\n"

        new_numerical_ordered_poiname = self.user_favorites.loc[new_numerical_order, "name"].tolist()
        return_candidates = [str(i) for i in range(len(new_numerical_ordered_poiname))]

        # 临时跳过选择起点的 API 调用，使用默认值
        if self.start_poi is not None and self.start_poi in new_numerical_ordered_poiname:
            response = [new_numerical_ordered_poiname.index(self.start_poi)]
        else:
            response = ["0"]  # 默认选择第一个点

        newnew_numerical_order = []
        newnew_numerical_order.extend(new_numerical_order[int(response[0]):])
        newnew_numerical_order.extend(new_numerical_order[:int(response[0])])
        new_numerical_order = newnew_numerical_order

        must_see_string, context_string = "", ""
        must_see_poi_idlist = self.must_see_pois
        if len(must_see_poi_idlist) > 0:
            must_see_poi_namelist = self.user_favorites.loc[must_see_poi_idlist, "name"].tolist()
            for name in self.must_see_poi_names:
                if name not in must_see_poi_namelist:
                    must_see_poi_namelist.append(name)
            must_see_string = str(must_see_poi_namelist)
        else:
            must_see_string = "No mandatory POIs selected"

        for i, (id, name, context) in enumerate(
                zip(new_numerical_order, self.user_favorites.loc[new_numerical_order, "name"],
                    self.user_favorites.loc[new_numerical_order, "context"])):
            new_context = f'Sequence number {i + 1}' + ':' + '"' + context.replace("\n", "")[:100] + '"'
            context_string += new_context + "\n"

        # 临时跳过检查反转的 API 调用，默认不反转
        response = ["0"]

        if int(response[0]) == 0:
            new_numerical_order.reverse()

        context_string = ""
        for i, (id, name, context) in enumerate(
                zip(new_numerical_order, self.user_favorites.loc[new_numerical_order, "name"],
                    self.user_favorites.loc[new_numerical_order, "context"])):
            new_context = f'Sequence number {i + 1}' + ':' + '"' + context.replace("\n", "")[:100] + '"'
            context_string += new_context + "\n"

        lookup_dict = {}
        for i, (id, name) in enumerate(zip(new_numerical_order, self.user_favorites.loc[new_numerical_order, "name"])):
            lookup_dict[i + 1] = [self.r2i[id], name]

        return context_string, must_see_string, lookup_dict, new_numerical_order

    def get_full_order(self, all_pairs: list, clusters: list, clusters_order: np.ndarray):
        new_numerical_order, lookup_dict = [], {}

        # 收集所有候选点
        for i, cluster_order in enumerate(clusters_order):
            cluster = clusters[cluster_order]
            new_numerical_order.extend(cluster)

        # 确保起点和终点
        start_idx, end_idx = None, None
        if self.start_poi:
            start_idx = self.user_favorites.index[self.user_favorites['name'] == self.start_poi].tolist()
            if start_idx and start_idx[0] not in new_numerical_order:
                new_numerical_order.insert(0, start_idx[0])
        if self.end_poi:
            end_idx = self.user_favorites.index[self.user_favorites['name'] == self.end_poi].tolist()
            if end_idx and end_idx[0] not in new_numerical_order:
                new_numerical_order.append(end_idx[0])

        # 移除重复点
        new_numerical_order = remove_duplicates(new_numerical_order)

        # 使用TSP优化路径
        if len(new_numerical_order) > 2:  # 至少有起点、一个中间点和终点
            # 提取经纬度
            points = self.user_favorites.loc[new_numerical_order][["lon", "lat"]].astype("float").to_numpy()
            # 计算距离矩阵（使用欧几里得距离，简化计算）
            dist_matrix = scipy.spatial.distance.cdist(points, points)

            # 固定起点和终点，优化中间点顺序
            start_pos = 0
            end_pos = len(new_numerical_order) - 1
            intermediate_indices = list(range(1, len(new_numerical_order) - 1))

            # 如果没有中间点，直接跳过TSP优化
            if not intermediate_indices:
                pass
            else:
                # 提取中间点的距离矩阵
                intermediate_dist_matrix = dist_matrix[intermediate_indices][:, intermediate_indices]
                # 使用TSP算法优化中间点顺序
                _, tsp_order = self.spatial_handler.solve_tsp_with_start_end(
                    intermediate_dist_matrix, 0, len(intermediate_indices) - 1
                )
                # 调整顺序
                reordered_intermediate = [intermediate_indices[i] for i in tsp_order[:-1]]  # 去掉重复的终点
                final_order = [new_numerical_order[0]] + [new_numerical_order[i] for i in reordered_intermediate] + [
                    new_numerical_order[-1]]
                new_numerical_order = final_order

        # 计算路径信息
        context_string, must_see_string, lookup_dict, new_numerical_order = self.calculate_ordered_route_info(
            new_numerical_order)

        # 生成完整的行程数据
        full_response_data = {}
        pois = {}
        itinerary_steps = []
        for i, poi_id in enumerate(new_numerical_order):
            poi_info = {
                'name': self.user_favorites.loc[poi_id, 'name'],
                'latitude': self.user_favorites.loc[poi_id, 'lat'],
                'longitude': self.user_favorites.loc[poi_id, 'lon'],
            }
            pois[str(i + 1)] = poi_info
            if i < len(new_numerical_order) - 1:
                name1 = self.user_favorites.loc[poi_id, 'name']
                name2 = self.user_favorites.loc[new_numerical_order[i + 1], 'name']
                if not name1 or not name2:
                    logging.warning(f"Missing name for POI {poi_id} or {new_numerical_order[i + 1]}")
                    continue
                step = f"From {name1} to {name2}"
                itinerary_steps.append(step)
        full_response_data['pois'] = pois
        full_response_data['itinerary'] = '->'.join(self.user_favorites.loc[new_numerical_order, 'name'].tolist())
        full_response_data['itinerary_steps'] = itinerary_steps
        return new_numerical_order, lookup_dict, clusters, context_string, must_see_string, full_response_data

    def get_day_plan(self, new_numerical_order, context_string, must_see_string):

        if not self.mark_citywalk and self.citywalk:
            comments = "To meet your needs, the planned route may require transportation. If a walking route is preferred, consider reducing requirements or adding more bookmarked locations."
        else:
            comments = ""

        messages = [
            {
                "role": "system",
                "content": get_system_prompt(self.maxPoiNum, len(self.must_see_pois), len(new_numerical_order))
            },
            {
                "role": "user",
                "content": get_dayplan_prompt(
                    context_string=context_string, must_see_string=must_see_string, keyword_reqs=self.user_pos_reqs,
                    userReqList=self.user_reqs,
                    maxPoiNum=self.maxPoiNum, numMustSee=len(self.must_see_pois),
                    numCandidates=len(new_numerical_order),
                    comments=comments, hours=self.hours, mark_citywalk=self.mark_citywalk,
                    itinerary_reqs=(self.itinerary_pos_reqs, self.itinerary_neg_reqs),
                    start_end=(self.start_poi, self.end_poi)
                )
            }
        ]

        return self.proxy.chat(messages=messages, model=self.MODEL, temperature=0)

    def solve(self):
        logging.info("Starting get_reqs_topk")
        req_topk_pois, pseudo_must_see_pois = self.get_reqs_topk()
        logging.info("Starting get_poi_candidates")
        all_pairs, clusters, clusters_order = self.get_poi_candidates(req_topk_pois, self.must_see_pois,
                                                                      pseudo_must_see_pois)
        logging.info("Starting get_full_order")
        new_numerical_order, lookup_dict, clusters, context_string, must_see_string, full_response_data = self.get_full_order(
            all_pairs, clusters, clusters_order)
        logging.info("Starting get_day_plan")
        full_response = self.get_day_plan(new_numerical_order, context_string, must_see_string)

        # 解析 get_day_plan 的结果
        try:
            itinerary_data = json.loads(full_response)
            selected_pois = itinerary_data["itinerary"].split("->")

            # 规范化起点和终点名称
            start_poi_normalized = self.start_poi.replace(" in Shanghai", "").strip()
            end_poi_normalized = self.end_poi.replace(" in Shanghai", "").strip()

            # 确保起点和终点正确
            available_pois = [poi for idx, poi in self.user_favorites.loc[new_numerical_order, "name"].items()]
            start_poi_matched = next((poi for poi in available_pois if poi == start_poi_normalized), None)
            end_poi_matched = next((poi for poi in available_pois if poi == end_poi_normalized), None)

            if not start_poi_matched or not end_poi_matched:
                logging.error(
                    f"Start or end point not found in available POIs: start={start_poi_normalized}, end={end_poi_normalized}")
                raise ValueError("Start or end point not found in available POIs")

            # 替换 selected_pois 中的起点和终点
            selected_pois = [start_poi_matched if poi == self.start_poi else poi for poi in selected_pois]
            selected_pois = [end_poi_matched if poi == self.end_poi else poi for poi in selected_pois]

            # 确保包含 art 和 museum 景点
            art_keywords = ["art", "gallery", "museum"]
            museum_keywords = ["museum", "memorial"]
            has_art = False
            has_museum = False
            for poi in selected_pois:
                poi_lower = poi.lower()
                if any(keyword in poi_lower for keyword in art_keywords):
                    has_art = True
                if any(keyword in poi_lower for keyword in museum_keywords):
                    has_museum = True
            if not has_art or not has_museum:
                logging.warning("Itinerary does not meet art and museum requirements. Adjusting...")
                art_poi = None
                museum_poi = None
                for idx, poi in self.user_favorites.loc[new_numerical_order, "name"].items():
                    poi_lower = poi.lower()
                    if not has_art and any(keyword in poi_lower for keyword in art_keywords):
                        art_poi = poi
                        has_art = True
                    if not has_museum and any(keyword in poi_lower for keyword in museum_keywords):
                        museum_poi = poi
                        has_museum = True
                if art_poi and art_poi not in selected_pois:
                    selected_pois.insert(-1, art_poi)
                if museum_poi and museum_poi not in selected_pois:
                    selected_pois.insert(-1, museum_poi)

            # 确保起点和终点正确
            if selected_pois[0] != start_poi_matched:
                selected_pois.insert(0, start_poi_matched)
            if selected_pois[-1] != end_poi_matched:
                selected_pois.append(end_poi_matched)

            # 移除重复点
            seen = set()
            unique_pois = []
            for poi in selected_pois:
                if poi not in seen:
                    seen.add(poi)
                    unique_pois.append(poi)
            selected_pois = unique_pois

            # 限制地点数量为 6-10 个
            if len(selected_pois) > 10:
                selected_pois = selected_pois[:5] + selected_pois[-5:]
            elif len(selected_pois) < 6:
                available_pois = [poi for idx, poi in self.user_favorites.loc[new_numerical_order, "name"].items() if
                                  poi not in selected_pois]
                for i in range(6 - len(selected_pois)):
                    if available_pois:
                        selected_pois.insert(-1, available_pois.pop(0))

            # 更新 full_response_data 的行程
            full_response_data["itinerary"] = "->".join(selected_pois)
            full_response_data["itinerary_steps"] = [f"From {selected_pois[i]} to {selected_pois[i + 1]}" for i in
                                                     range(len(selected_pois) - 1)]

            # 筛选 pois，只保留 get_day_plan 选择的地点
            selected_indices = [str(i + 1) for i, poi in enumerate(self.user_favorites.loc[new_numerical_order, "name"])
                                if poi in selected_pois]
            full_response_data["pois"] = {k: v for k, v in full_response_data["pois"].items() if k in selected_indices}
        except json.JSONDecodeError:
            logging.error(f"Failed to parse get_day_plan response: {full_response}")
            # 回退逻辑：生成默认行程
            selected_pois = [start_poi_normalized]
            art_poi = None
            museum_poi = None
            for idx, poi in self.user_favorites.loc[new_numerical_order, "name"].items():
                poi_lower = poi.lower()
                if not art_poi and any(keyword in poi_lower for keyword in ["art", "gallery", "museum"]):
                    art_poi = poi
                if not museum_poi and any(keyword in poi_lower for keyword in ["museum", "memorial"]):
                    museum_poi = poi
            if art_poi and art_poi != start_poi_normalized and art_poi != end_poi_normalized:
                selected_pois.append(art_poi)
            if museum_poi and museum_poi != start_poi_normalized and museum_poi != end_poi_normalized and museum_poi != art_poi:
                selected_pois.append(museum_poi)
            available_pois = [poi for idx, poi in self.user_favorites.loc[new_numerical_order, "name"].items() if
                              poi not in selected_pois and poi != end_poi_normalized]
            for i in range(min(7 - len(selected_pois), len(available_pois))):
                selected_pois.append(available_pois[i])
            selected_pois.append(end_poi_normalized)
            # 移除重复点
            seen = set()
            unique_pois = []
            for poi in selected_pois:
                if poi not in seen:
                    seen.add(poi)
                    unique_pois.append(poi)
            selected_pois = unique_pois
            # 更新 full_response_data
            full_response_data["itinerary"] = "->".join(selected_pois)
            full_response_data["itinerary_steps"] = [f"From {selected_pois[i]} to {selected_pois[i + 1]}" for i in
                                                     range(len(selected_pois) - 1)]
            selected_indices = [str(i + 1) for i, poi in enumerate(self.user_favorites.loc[new_numerical_order, "name"])
                                if poi in selected_pois]
            full_response_data["pois"] = {k: v for k, v in full_response_data["pois"].items() if k in selected_indices}

        logging.info("Finished solving itinerary")
        print(f'Itinerary: \n{full_response_data}')
        return full_response_data, lookup_dict

''' 
    def solve(self):
        """Main function to compute the day plan based on requirements and preferences."""
        logging.info("Starting get_reqs_topk")
        req_topk_pois, pseudo_must_see_pois = self.get_reqs_topk()
        logging.info("Starting get_poi_candidates")
        all_pairs, clusters, clusters_order = self.get_poi_candidates(req_topk_pois, self.must_see_pois,
                                                                      pseudo_must_see_pois)
        logging.info("Starting get_full_order")
        new_numerical_order, lookup_dict, clusters, context_string, must_see_string, full_response_data = self.get_full_order(
            all_pairs, clusters, clusters_order)
        logging.info("Starting get_day_plan")
        full_response = self.get_day_plan(new_numerical_order, context_string, must_see_string)
        logging.info("Finished solving itinerary")
        # 这里不再需要重复调用 save_qualitative，因为 get_full_order 已经返回了 full_response_data

        # 检查生成的poi信息中的坐标
        pois = full_response_data.get('pois', {})
        for poi_key, poi in pois.items():
            if isinstance(poi, dict):
                lat = poi.get('latitude')
                lng = poi.get('longitude')
                if lat is None or lng is None:
                    logging.warning('Missing latitude or longitude in poi with key %s: %s', poi_key, poi)
                else:
                    try:
                        lat = float(lat)
                        lng = float(lng)
                        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                            logging.warning('Invalid latitude or longitude in poi with key %s: %s', poi_key, poi)
                    except ValueError:
                        logging.warning('Invalid latitude or longitude value in poi with key %s: %s', poi_key, poi)

        print(f'Itinerary: \n{full_response_data}')
        return full_response_data, lookup_dict
'''