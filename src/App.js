import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Layout, Input, Button, List } from 'antd';
import 'antd/dist/reset.css';
import './App.css';

const { Content } = Layout;

const App = () => {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [itinerary, setItinerary] = useState([]); // 保留 itinerary 状态
    const [itinerarySteps, setItinerarySteps] = useState([]);
    const [spots, setSpots] = useState([]);
    const [error, setError] = useState('');

    useEffect(() => {
        if (window.AMap) {
            const map = new window.AMap.Map('map', {
                zoom: 13,
                center: [121.4737, 31.2304],
                viewMode: '2D',
                resizeEnable: true,
            });
            map.addControl(new window.AMap.Scale());
            map.addControl(new window.AMap.ToolBar());
        }
    }, []);

const generateItinerary = async () => {
    if (!query) {
        setError('Please enter your itinerary request');
        return;
    }

    setLoading(true);
    setError('');
    setItinerary([]);
    setItinerarySteps([]);
    setSpots([]);

    try {
        const response = await axios.post('http://127.0.0.1:5000/generate_itinerary', {
            request: query,
            city: 'shanghai'
        });

        const { itinerary, itinerary_steps, spots } = response.data;
        console.log('Received itinerary:', itinerary);
        console.log('Itinerary steps:', itinerary_steps);
        console.log('Spots:', spots);

        setItinerary(itinerary);
        setItinerarySteps(itinerary_steps);
        setSpots(spots);

        if (window.AMap && spots.length > 0) {
            const map = new window.AMap.Map('map', {
                zoom: 13,
                center: [spots[0].longitude, spots[0].latitude],
                viewMode: '2D',
                resizeEnable: true
            });

            const markers = spots.map((spot, index) => {
                const marker = new window.AMap.Marker({
                    position: [spot.longitude, spot.latitude],
                    title: spot.name
                });

                const infoWindow = new window.AMap.InfoWindow({
                    content: `
                        <div>
                            <h4>${index + 1}. ${spot.name}</h4>
                            <p>${spot.address || ''}</p>
                        </div>
                    `,
                    offset: new window.AMap.Pixel(0, -30)
                });

                marker.on('click', () => {
                    infoWindow.open(map, marker.getPosition());
                });

                marker.setMap(map);
                return marker;
            });

            // 手动绘制路径，避免覆盖
            const walking = new window.AMap.Walking({
                // 移除 map: map，避免自动绘制覆盖
                strokeColor: '#ff0000',
                strokeWeight: 5
            });

            const promises = [];
            const polylines = []; // 存储每段路径的折线

            for (let i = 0; i < spots.length - 1; i++) {
                const origin = [spots[i].longitude, spots[i].latitude];
                const destination = [spots[i + 1].longitude, spots[i + 1].latitude];
                console.log(`Planning path from ${spots[i].name} (${origin}) to ${spots[i + 1].name} (${destination})`);

                promises.push(
                    new Promise((resolve, reject) => {
                        walking.search(origin, destination, (status, result) => {
                            if (status === 'complete') {
                                console.log(`Path successfully planned from ${spots[i].name} to ${spots[i + 1].name}`);
                                console.log('Path result:', result);

                                // 手动绘制路径
                                const path = result.routes[0].steps.reduce((acc, step) => {
                                    return acc.concat(step.path);
                                }, []);
                                const polyline = new window.AMap.Polyline({
                                    path: path,
                                    strokeColor: '#ff0000',
                                    strokeWeight: 5,
                                    map: map
                                });
                                polylines.push(polyline);

                                resolve(result);
                            } else {
                                console.error(`Walking route search failed for ${spots[i].name} to ${spots[i + 1].name}: ${status}`);
                                console.error('Error details:', result);
                                reject(new Error(`Walking route search failed: ${status}`));
                            }
                        });
                    })
                );
            }

            const results = await Promise.allSettled(promises);
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    console.log(`Path ${index + 1} successfully drawn`);
                } else {
                    console.error(`Path ${index + 1} failed: ${result.reason}`);
                }
            });

            map.setFitView(markers);
        }
    } catch (error) {
        console.error('Error generating itinerary:', error);
        if (error.response) {
            setError(`Server returned error: ${error.response.statusText}`);
        } else if (error.request) {
            setError('No response received from server');
        } else {
            setError('Error occurred while generating itinerary');
        }
    } finally {
        setLoading(false);
    }
};

    return (
        <Layout>
            <Content style={{ padding: '24px' }}>
                <div className="container">
                    <div className="input-container">
                        <Input
                            placeholder="Please enter your itinerary request"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onPressEnter={generateItinerary}
                        />
                        <Button type="primary" onClick={generateItinerary} loading={loading}>
                            Generate Itinerary
                        </Button>
                    </div>

                    {error && <div style={{ color: 'red', marginBottom: '16px' }}>{error}</div>}

                    <div className="itinerary-map-container">
                        <div className="itinerary-steps">
                            {/* 显示 Itinerary Steps */}
                            {itinerarySteps.length > 0 && (
                                <div>
                                    <h2>Itinerary Steps:</h2>
                                    <List
                                        dataSource={itinerarySteps}
                                        renderItem={(item, index) => (
                                            <List.Item>
                                                {index + 1}. {item}
                                            </List.Item>
                                        )}
                                        locale={{ emptyText: 'No itinerary generated yet. Enter a request to get started!' }}
                                    />
                                </div>
                            )}

                            {/* 新增：显示简单的地点列表 */}
                            {itinerary.length > 0 && (
                                <div style={{ marginTop: '24px' }}>
                                    <h3>Itinerary Overview:</h3>
                                    <List
                                        dataSource={itinerary}
                                        renderItem={(item, index) => (
                                            <List.Item>
                                                {index + 1}. {item}
                                            </List.Item>
                                        )}
                                    />
                                </div>
                            )}

                            {/* 显示 Detailed Spots Information */}
                            {spots.length > 0 && (
                                <div className="spots-container">
                                    <h3>Detailed Spots Information:</h3>
                                    {spots.map((spot, index) => (
                                        <div key={index} className="spot-item">
                                            <div className="spot-info">
                                                <div className="spot-name">
                                                    {index + 1}. {spot.name}
                                                </div>
                                                <div className="spot-details">
                                                    <div>Address: {spot.address || 'N/A'}</div>
                                                    <div>Category: {spot.category || 'N/A'}</div>
                                                    <div>Rating: {spot.rating || 'N/A'}</div>
                                                    {spot.description && <div>Description: {spot.description}</div>}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div id="map" className="map-container"></div>
                    </div>
                </div>
            </Content>
        </Layout>
    );
};

export default App;