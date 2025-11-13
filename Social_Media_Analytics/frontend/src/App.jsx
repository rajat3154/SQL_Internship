import React, { useState, useEffect } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_BACKEND_URL;

function App() {
  const [activeTab, setActiveTab] = useState("analytics");
  const [users, setUsers] = useState([]);
  const [posts, setPosts] = useState([]);
  const [topPosts, setTopPosts] = useState([]);
  const [userSummary, setUserSummary] = useState([]);
  const [engagementStats, setEngagementStats] = useState({});
  const [searchResults, setSearchResults] = useState([]);
  const [unionActivities, setUnionActivities] = useState([]);
  const [groupedData, setGroupedData] = useState([]);

  const [newUser, setNewUser] = useState({
    username: "",
    email: "",
    full_name: "",
  });
  const [newPost, setNewPost] = useState({ user_id: "", content: "" });
  const [newLike, setNewLike] = useState({ post_id: "", user_id: "" });
  const [newComment, setNewComment] = useState({
    post_id: "",
    user_id: "",
    content: "",
  });
  const [searchQuery, setSearchQuery] = useState("");

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Fetch data based on active tab
  useEffect(() => {
    switch (activeTab) {
      case "users":
        fetchUsers();
        break;
      case "posts":
        fetchPosts();
        fetchUsers();
        break;
      case "analytics":
        fetchAnalytics();
        break;
      case "search":
        fetchUnionActivities();
        fetchGroupedData();
        break;
      default:
        break;
    }
  }, [activeTab]);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/users/`);
      setUsers(res.data);
    } catch (err) {
      setMessage("Error fetching users");
    } finally {
      setLoading(false);
    }
  };

  const fetchPosts = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/posts/`);
      setPosts(res.data);
    } catch (err) {
      setMessage("Error fetching posts");
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const [topRes, summaryRes, statsRes] = await Promise.all([
        axios.get(`${API_URL}/analytics/top-posts`),
        axios.get(`${API_URL}/analytics/user-summary`),
        axios.get(`${API_URL}/analytics/engagement-stats`),
      ]);
      setTopPosts(topRes.data);
      setUserSummary(summaryRes.data);
      setEngagementStats(statsRes.data);
    } catch (err) {
      setMessage("Error fetching analytics");
    } finally {
      setLoading(false);
    }
  };

  const fetchUnionActivities = async () => {
    try {
      const res = await axios.get(`${API_URL}/analytics/union-activities`);
      setUnionActivities(res.data);
    } catch (err) {
      setMessage("Error fetching activities");
    }
  };

  const fetchGroupedData = async () => {
    try {
      const res = await axios.get(`${API_URL}/analytics/group-by-engagement`);
      setGroupedData(res.data);
    } catch (err) {
      setMessage("Error fetching grouped data");
    }
  };

  const createUser = async () => {
    if (!newUser.username || !newUser.email || !newUser.full_name) {
      setMessage("Please fill all fields");
      return;
    }
    setLoading(true);
    try {
      await axios.post(`${API_URL}/users/`, newUser);
      setMessage("User created successfully!");
      setNewUser({ username: "", email: "", full_name: "" });
      fetchUsers();
    } catch (err) {
      setMessage("Error creating user");
    } finally {
      setLoading(false);
    }
  };

  const createPost = async () => {
    if (!newPost.user_id || !newPost.content) {
      setMessage("Please fill all fields");
      return;
    }
    setLoading(true);
    try {
      await axios.post(`${API_URL}/posts/`, newPost);
      setMessage("Post created successfully!");
      setNewPost({ user_id: "", content: "" });
      fetchPosts();
      fetchAnalytics();
    } catch (err) {
      setMessage("Error creating post");
    } finally {
      setLoading(false);
    }
  };

  const addLike = async () => {
    if (!newLike.post_id || !newLike.user_id) {
      setMessage("Please fill all fields");
      return;
    }
    try {
      await axios.post(`${API_URL}/likes/`, newLike);
      setMessage("Like added successfully!");
      setNewLike({ post_id: "", user_id: "" });
      fetchAnalytics();
    } catch (err) {
      setMessage("Error adding like");
    }
  };

  const addComment = async () => {
    if (!newComment.post_id || !newComment.user_id || !newComment.content) {
      setMessage("Please fill all fields");
      return;
    }
    try {
      await axios.post(`${API_URL}/comments/`, newComment);
      setMessage("Comment added successfully!");
      setNewComment({ post_id: "", user_id: "", content: "" });
      fetchAnalytics();
    } catch (err) {
      setMessage("Error adding comment");
    }
  };

  const searchPosts = async () => {
    if (!searchQuery) {
      setMessage("Please enter search query");
      return;
    }
    try {
      const res = await axios.get(
        `${API_URL}/analytics/search-posts?query=${searchQuery}`
      );
      setSearchResults(res.data);
      setMessage(`Found ${res.data.length} posts`);
    } catch (err) {
      setMessage("Error searching posts");
    }
  };

  const refreshMaterialized = async () => {
    try {
      await axios.post(`${API_URL}/analytics/refresh-materialized`);
      setMessage("Views refreshed successfully!");
      fetchAnalytics();
    } catch (err) {
      setMessage("Error refreshing views");
    }
  };

  const exportReport = async () => {
    try {
      const response = await axios.get(`${API_URL}/analytics/export-report`, {
        responseType: "blob",
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "engagement_report.csv");
      document.body.appendChild(link);
      link.click();
      link.remove();
      setMessage("Report exported successfully!");
    } catch (err) {
      setMessage("Error exporting report");
    }
  };

  const deleteUser = async (userId) => {
    if (window.confirm("Are you sure you want to delete this user?")) {
      try {
        await axios.delete(`${API_URL}/users/${userId}`);
        setMessage("User deleted successfully!");
        fetchUsers();
      } catch (err) {
        setMessage("Error deleting user");
      }
    }
  };

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">
            📊 Social Media Analytics Dashboard
          </h1>
          <p className="text-blue-300 text-lg">
            Complete CRUD operations with advanced SQL features
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-gray-950 rounded-xl p-2 mb-8 border border-blue-500/20">
          <div className="flex space-x-2">
            {["analytics", "users", "posts", "search", "actions"].map((tab) => (
              <button
                key={tab}
                className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all duration-200 ${
                  activeTab === tab
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-500/25"
                    : "text-gray-300 hover:bg-black hover:text-white"
                }`}
                onClick={() => setActiveTab(tab)}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Message Alert */}
        {message && (
          <div
            className={`mb-6 p-4 rounded-lg border ${
              message.includes("Error")
                ? "bg-red-900/20 border-red-500 text-red-200"
                : "bg-green-900/20 border-green-500 text-green-200"
            }`}
          >
            {message}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="text-center py-8">
            <div className="inline-flex items-center px-4 py-2 bg-blue-600 rounded-lg text-white">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-3"></div>
              Loading...
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === "analytics" && (
          <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {engagementStats.overall_stats && (
                <>
                  <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                    <div className="text-blue-400 text-sm font-semibold mb-2">
                      Total Posts
                    </div>
                    <div className="text-3xl font-bold text-white">
                      {engagementStats.overall_stats.total_posts}
                    </div>
                  </div>
                  <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                    <div className="text-blue-400 text-sm font-semibold mb-2">
                      Total Likes
                    </div>
                    <div className="text-3xl font-bold text-white">
                      {engagementStats.overall_stats.total_likes}
                    </div>
                  </div>
                  <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                    <div className="text-blue-400 text-sm font-semibold mb-2">
                      Total Comments
                    </div>
                    <div className="text-3xl font-bold text-white">
                      {engagementStats.overall_stats.total_comments}
                    </div>
                  </div>
                  <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                    <div className="text-blue-400 text-sm font-semibold mb-2">
                      Avg Engagement
                    </div>
                    <div className="text-3xl font-bold text-white">
                      {engagementStats.overall_stats.avg_engagement?.toFixed(2)}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Analytics Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Top Posts */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                  🏆 Top Posts (Window Functions)
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Rank
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Username
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Content
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Likes
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Engagement
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {topPosts.map((post) => (
                        <tr
                          key={post.post_id}
                          className="border-b border-gray-700/50 hover:bg-black/30"
                        >
                          <td className="py-3 px-4 text-white font-semibold">
                            {post.rank}
                          </td>
                          <td className="py-3 px-4 text-gray-300">
                            {post.username}
                          </td>
                          <td className="py-3 px-4 text-gray-300 max-w-xs truncate">
                            {post.content}
                          </td>
                          <td className="py-3 px-4 text-white">
                            {post.like_count}
                          </td>
                          <td className="py-3 px-4 text-blue-400 font-semibold">
                            {post.engagement_score?.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* User Summary */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                  👥 User Summary (GROUP BY)
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Username
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Posts
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Likes Received
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Comments Received
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {userSummary.map((user) => (
                        <tr
                          key={user.user_id}
                          className="border-b border-gray-700/50 hover:bg-black/30"
                        >
                          <td className="py-3 px-4 text-white font-semibold">
                            {user.username}
                          </td>
                          <td className="py-3 px-4 text-gray-300">
                            {user.total_posts}
                          </td>
                          <td className="py-3 px-4 text-gray-300">
                            {user.total_likes_received}
                          </td>
                          <td className="py-3 px-4 text-gray-300">
                            {user.total_comments_received}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Users Tab */}
        {activeTab === "users" && (
          <div className="space-y-6">
            {/* Create User Form */}
            <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-bold text-white mb-4">
                Create New User
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <input
                  type="text"
                  placeholder="Username"
                  value={newUser.username}
                  onChange={(e) =>
                    setNewUser({ ...newUser, username: e.target.value })
                  }
                  className="bg-black border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
                <input
                  type="email"
                  placeholder="Email"
                  value={newUser.email}
                  onChange={(e) =>
                    setNewUser({ ...newUser, email: e.target.value })
                  }
                  className="bg-black border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
                <input
                  type="text"
                  placeholder="Full Name"
                  value={newUser.full_name}
                  onChange={(e) =>
                    setNewUser({ ...newUser, full_name: e.target.value })
                  }
                  className="bg-black border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
                <button
                  onClick={createUser}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                >
                  Create User
                </button>
              </div>
            </div>

            {/* Users Table */}
            <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-bold text-white mb-4">All Users</h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        ID
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Username
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Email
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Full Name
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Created
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((user) => (
                      <tr
                        key={user.id}
                        className="border-b border-gray-700/50 hover:bg-black/30"
                      >
                        <td className="py-3 px-4 text-white font-semibold">
                          {user.id}
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {user.username}
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {user.email}
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {user.full_name}
                        </td>
                        <td className="py-3 px-4 text-gray-400 text-sm">
                          {new Date(user.created_at).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4">
                          <button
                            onClick={() => deleteUser(user.id)}
                            className="bg-red-600 hover:bg-red-700 text-white text-sm font-semibold rounded px-3 py-1 transition-colors"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Posts Tab */}
        {activeTab === "posts" && (
          <div className="space-y-6">
            {/* Action Forms Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Create Post */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4">
                  Create Post
                </h2>
                <div className="space-y-4">
                  <select
                    value={newPost.user_id}
                    onChange={(e) =>
                      setNewPost({ ...newPost, user_id: e.target.value })
                    }
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="" className="bg-black">
                      Select User
                    </option>
                    {users.map((user) => (
                      <option
                        key={user.id}
                        value={user.id}
                        className="bg-black"
                      >
                        {user.username}
                      </option>
                    ))}
                  </select>
                  <textarea
                    placeholder="Post content"
                    value={newPost.content}
                    onChange={(e) =>
                      setNewPost({ ...newPost, content: e.target.value })
                    }
                    rows="3"
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                  <button
                    onClick={createPost}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                  >
                    Create Post
                  </button>
                </div>
              </div>

              {/* Add Like */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4">Add Like</h2>
                <div className="space-y-4">
                  <select
                    value={newLike.user_id}
                    onChange={(e) =>
                      setNewLike({ ...newLike, user_id: e.target.value })
                    }
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="" className="bg-black">
                      Select User
                    </option>
                    {users.map((user) => (
                      <option
                        key={user.id}
                        value={user.id}
                        className="bg-black"
                      >
                        {user.username}
                      </option>
                    ))}
                  </select>
                  <select
                    value={newLike.post_id}
                    onChange={(e) =>
                      setNewLike({ ...newLike, post_id: e.target.value })
                    }
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="" className="bg-black">
                      Select Post
                    </option>
                    {posts.map((post) => (
                      <option
                        key={post.id}
                        value={post.id}
                        className="bg-black"
                      >
                        Post #{post.id} by User #{post.user_id}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={addLike}
                    className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                  >
                    Add Like
                  </button>
                </div>
              </div>

              {/* Add Comment */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4">
                  Add Comment
                </h2>
                <div className="space-y-4">
                  <select
                    value={newComment.user_id}
                    onChange={(e) =>
                      setNewComment({ ...newComment, user_id: e.target.value })
                    }
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="" className="bg-black">
                      Select User
                    </option>
                    {users.map((user) => (
                      <option
                        key={user.id}
                        value={user.id}
                        className="bg-black"
                      >
                        {user.username}
                      </option>
                    ))}
                  </select>
                  <select
                    value={newComment.post_id}
                    onChange={(e) =>
                      setNewComment({ ...newComment, post_id: e.target.value })
                    }
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="" className="bg-black">
                      Select Post
                    </option>
                    {posts.map((post) => (
                      <option
                        key={post.id}
                        value={post.id}
                        className="bg-black"
                      >
                        Post #{post.id} by User #{post.user_id}
                      </option>
                    ))}
                  </select>
                  <textarea
                    placeholder="Comment content"
                    value={newComment.content}
                    onChange={(e) =>
                      setNewComment({ ...newComment, content: e.target.value })
                    }
                    rows="2"
                    className="w-full bg-black border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                  <button
                    onClick={addComment}
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                  >
                    Add Comment
                  </button>
                </div>
              </div>
            </div>

            {/* Posts Table */}
            <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-bold text-white mb-4">All Posts</h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        ID
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        User ID
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Content
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Likes
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Comments
                      </th>
                      <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                        Engagement
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {posts.map((post) => (
                      <tr
                        key={post.id}
                        className="border-b border-gray-700/50 hover:bg-black/30"
                      >
                        <td className="py-3 px-4 text-white font-semibold">
                          {post.id}
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {post.user_id}
                        </td>
                        <td className="py-3 px-4 text-gray-300 max-w-md">
                          {post.content}
                        </td>
                        <td className="py-3 px-4 text-green-400">
                          {post.like_count}
                        </td>
                        <td className="py-3 px-4 text-blue-400">
                          {post.comment_count}
                        </td>
                        <td className="py-3 px-4 text-yellow-400 font-semibold">
                          {post.engagement_score?.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Search Tab */}
        {activeTab === "search" && (
          <div className="space-y-6">
            {/* Search Section */}
            <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-bold text-white mb-4">
                Search Posts (LIKE operator)
              </h2>
              <div className="flex gap-4">
                <input
                  type="text"
                  placeholder="Search post content..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1 bg-black border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
                <button
                  onClick={searchPosts}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg px-8 py-3 transition-colors"
                >
                  Search
                </button>
              </div>

              {searchResults.length > 0 && (
                <div className="mt-6 overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Username
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Content
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Likes
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Engagement
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {searchResults.map((post) => (
                        <tr
                          key={post.id}
                          className="border-b border-gray-700/50 hover:bg-black/30"
                        >
                          <td className="py-3 px-4 text-white font-semibold">
                            {post.username}
                          </td>
                          <td className="py-3 px-4 text-gray-300">
                            {post.content}
                          </td>
                          <td className="py-3 px-4 text-green-400">
                            {post.like_count}
                          </td>
                          <td className="py-3 px-4 text-yellow-400">
                            {post.engagement_score?.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Analytics Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Union Activities */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4">
                  Union Activities (UNION ALL)
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Activity
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Username
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Content
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Date
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {unionActivities.map((activity, index) => (
                        <tr
                          key={index}
                          className="border-b border-gray-700/50 hover:bg-black/30"
                        >
                          <td className="py-3 px-4">
                            <span
                              className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-semibold ${
                                activity.activity_type === "POST"
                                  ? "bg-blue-500/20 text-blue-400"
                                  : activity.activity_type === "LIKE"
                                  ? "bg-green-500/20 text-green-400"
                                  : "bg-purple-500/20 text-purple-400"
                              }`}
                            >
                              {activity.activity_type}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-white font-semibold">
                            {activity.username}
                          </td>
                          <td className="py-3 px-4 text-gray-300 max-w-xs truncate">
                            {activity.content}
                          </td>
                          <td className="py-3 px-4 text-gray-400 text-sm">
                            {new Date(activity.activity_date).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Grouped Data */}
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-4">
                  Grouped Engagement (GROUP BY + HAVING)
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Username
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Post Count
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Avg Engagement
                        </th>
                        <th className="text-left py-3 px-4 text-blue-400 font-semibold">
                          Engagement Level
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {groupedData.map((user, index) => (
                        <tr
                          key={index}
                          className="border-b border-gray-700/50 hover:bg-black/30"
                        >
                          <td className="py-3 px-4 text-white font-semibold">
                            {user.username}
                          </td>
                          <td className="py-3 px-4 text-gray-300">
                            {user.post_count}
                          </td>
                          <td className="py-3 px-4 text-yellow-400 font-semibold">
                            {user.avg_engagement?.toFixed(2)}
                          </td>
                          <td className="py-3 px-4">
                            <span
                              className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold ${
                                user.engagement_level === "High"
                                  ? "bg-green-500/20 text-green-400"
                                  : user.engagement_level === "Medium"
                                  ? "bg-yellow-500/20 text-yellow-400"
                                  : "bg-red-500/20 text-red-400"
                              }`}
                            >
                              {user.engagement_level}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Actions Tab */}
        {activeTab === "actions" && (
          <div className="space-y-6">
            {/* Actions Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                <div className="text-3xl mb-4">🔄</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  Refresh Materialized Views
                </h3>
                <p className="text-gray-400 mb-4">
                  Refresh materialized views and stored procedures
                </p>
                <button
                  onClick={refreshMaterialized}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                >
                  Refresh Views
                </button>
              </div>

              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                <div className="text-3xl mb-4">📊</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  Export Analytics Report
                </h3>
                <p className="text-gray-400 mb-4">
                  Download comprehensive engagement report as CSV
                </p>
                <button
                  onClick={exportReport}
                  className="bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                >
                  Export Report
                </button>
              </div>

              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                <div className="text-3xl mb-4">⚡</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  Trigger Test
                </h3>
                <p className="text-gray-400 mb-2">
                  Test database triggers by adding engagement
                </p>
                <p className="text-blue-400 text-sm">
                  Triggers automatically update counts and scores
                </p>
              </div>

              <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 text-center shadow-lg">
                <div className="text-3xl mb-4">📈</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  Stored Procedure
                </h3>
                <p className="text-gray-400 mb-4">
                  Refresh engagement metrics using stored procedure
                </p>
                <button
                  onClick={refreshMaterialized}
                  className="bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-lg px-6 py-3 transition-colors"
                >
                  Run Procedure
                </button>
              </div>
            </div>

            {/* Features List */}
            <div className="bg-gray-950 border border-blue-500/20 rounded-xl p-6 shadow-lg">
              <h2 className="text-2xl font-bold text-white mb-6 text-center">
                ✅ Implemented SQL Features
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {[
                  "CRUD Operations",
                  "GROUP BY with HAVING",
                  "ORDER BY with sorting",
                  "TRIGGERS (auto-update counts)",
                  "UNION/INTERSECT operations",
                  "LIKE pattern matching",
                  "IN keyword filtering",
                  "WINDOW FUNCTIONS (RANK)",
                  "STORED PROCEDURES",
                  "MATERIALIZED VIEWS",
                  "VIEWS for abstraction",
                  "JOIN operations",
                ].map((feature, index) => (
                  <div
                    key={index}
                    className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 text-center"
                  >
                    <div className="text-blue-400 font-semibold">{feature}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
