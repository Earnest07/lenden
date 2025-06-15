// // script.js (Updated for Unified Backend)

// // --- CONFIGURATION FOR ALL POSSIBLE INPUT FEATURES ACROSS ALL MODELS ---
// // This comprehensive list is used to build the single payload sent to the unified backend.
// const ALL_FEATURES_CONFIG = [
//   // Demographic Information
//   {
//     name: "age",
//     label: "Age (Numerical)",
//     type: "number",
//     category: "Demographic Information",
//   },
//   {
//     name: "gender",
//     label: "Gender (Categorical)",
//     type: "select",
//     options: ["Male", "Female", "Other"],
//     category: "Demographic Information",
//   },
//   {
//     name: "marital_status",
//     label: "Marital Status (Categorical)",
//     type: "select",
//     options: ["Married", "Single", "Divorced", "Widowed"],
//     category: "Demographic Information",
//   },
//   {
//     name: "residence_ownership",
//     label: "Residence Ownership (Categorical)",
//     type: "select",
//     options: ["Owned", "Rented", "Jointly Owned"],
//     category: "Demographic Information",
//   },
//   {
//     name: "city",
//     label: "City (Categorical)",
//     type: "text",
//     category: "Location Information",
//   },
//   {
//     name: "state",
//     label: "State (Categorical)",
//     type: "text",
//     category: "Location Information",
//   },
//   {
//     name: "pin",
//     label: "PIN Code (Categorical)",
//     type: "text",
//     category: "Location Information",
//   },

//   // Device Information
//   {
//     name: "device_model",
//     label: "Device Model (Categorical)",
//     type: "text",
//     category: "Device Information",
//   },
//   {
//     name: "device_category",
//     label: "Device Category (Categorical)",
//     type: "select",
//     options: ["Smartphone", "Tablet", "Laptop", "Desktop"],
//     category: "Device Information",
//   },
//   {
//     name: "platform",
//     label: "Platform (Categorical)",
//     type: "select",
//     options: ["iOS", "Android", "Windows", "macOS", "Linux"],
//     category: "Device Information",
//   },
//   {
//     name: "device_manufacturer",
//     label: "Device Manufacturer (Categorical)",
//     type: "text",
//     category: "Device Information",
//   },

//   // Financial/Behavioral Metrics (var_0 to var_73)
//   {
//     name: "var_0",
//     label: "Balance Var 0",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_1",
//     label: "Balance Var 1",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_2",
//     label: "Credit Limit Var 2",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_3",
//     label: "Credit Limit Var 3",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_4",
//     label: "Balance Var 4",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_5",
//     label: "Credit Limit Var 5",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_6",
//     label: "Loan Amount Var 6",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_7",
//     label: "Loan Amount Var 7",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_8",
//     label: "Balance Var 8",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_9",
//     label: "EMI Var 9",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_10",
//     label: "Credit Limit Var 10",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_11",
//     label: "Credit Limit Var 11",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_12",
//     label: "Credit Limit Var 12",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_13",
//     label: "Loan Amount Var 13",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_14",
//     label: "Loan Amount Var 14",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_15",
//     label: "Inquiry Var 15",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_16",
//     label: "Inquiry Var 16",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_17",
//     label: "EMI Var 17",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_18",
//     label: "Balance Var 18",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_19",
//     label: "Balance Var 19",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_20",
//     label: "Loan Amount Var 20",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_21",
//     label: "Balance Var 21",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_22",
//     label: "Credit Limit Var 22",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_23",
//     label: "Credit Limit Var 23",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_24",
//     label: "Loan Amount Var 24",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_25",
//     label: "Inquiry Var 25",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_26",
//     label: "Credit Limit Var 26",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_27",
//     label: "Credit Limit Var 27",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_28",
//     label: "Credit Limit Var 28",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_29",
//     label: "Credit Limit Var 29",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_30",
//     label: "Balance Var 30",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_31",
//     label: "Loan Amount Var 31",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_32",
//     label: "Credit Score",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_33",
//     label: "Credit Limit Var 33",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_34",
//     label: "Balance Var 34",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_35",
//     label: "Balance Var 35",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_36",
//     label: "Loan Amount Var 36",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_37",
//     label: "Repayment Var 37",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_38",
//     label: "Balance Var 38",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_39",
//     label: "Loan Amount Var 39",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_40",
//     label: "Closed Loans",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_41",
//     label: "EMI Var 41",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_42",
//     label: "Loan Amount Var 42",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_43",
//     label: "EMI Var 43",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_44",
//     label: "Credit Limit Var 44",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_45",
//     label: "Inquiry Var 45",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_46",
//     label: "EMI Var 46",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_47",
//     label: "Credit Limit Var 47",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_48",
//     label: "Repayment Var 48",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_49",
//     label: "Repayment Var 49",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_50",
//     label: "Repayment Var 50",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_51",
//     label: "EMI Var 51",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_52",
//     label: "Repayment Var 52",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_53",
//     label: "Loan Activity Var 53",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_54",
//     label: "Loan Activity Var 54",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_55",
//     label: "Repayment Var 55",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_56",
//     label: "EMI Var 56",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_57",
//     label: "Loan Activity Var 57",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_58",
//     label: "Inquiry Var 58",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_59",
//     label: "Balance Var 59",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_60",
//     label: "Loan Activity Var 60",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_61",
//     label: "Inquiry Var 61",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_62",
//     label: "Loan Activity Var 62",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_63",
//     label: "Loan Activity Var 63",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_64",
//     label: "Loan Activity Var 64",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_65",
//     label: "Loan Amount Var 65",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_66",
//     label: "Loan Activity Var 66",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_67",
//     label: "Repayment Var 67",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_68",
//     label: "Balance Var 68",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_69",
//     label: "Repayment Var 69",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_70",
//     label: "Repayment Var 70",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_71",
//     label: "Inquiry Var 71",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_72",
//     label: "Loan Amount Var 72",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_73",
//     label: "Repayment Var 73",
//     type: "number",
//     category: "Financial/Behavioral Metrics",
//   },
//   {
//     name: "var_74",
//     label: "Score Comments",
//     type: "select",
//     options: [
//       "Good Payment History",
//       "Late Payments",
//       "High Credit Utilization",
//       "Low Credit Score",
//       "No Credit History",
//       "Bankruptcies",
//       "Defaults",
//     ],
//     category: "Credit Score Related",
//   },
//   {
//     name: "var_75",
//     label: "Score Type",
//     type: "select",
//     options: ["A", "B", "C", "D", "E", "F"],
//     category: "Credit Score Related",
//   },

//   // Engineered Features expected by Creditworthiness (could be outputs from Master or direct inputs)
//   {
//     name: "final_predicted_income",
//     label: "Final Predicted Income (Creditworthiness Input)",
//     type: "number",
//     category: "Engineered/Derived Features",
//   },
//   {
//     name: "financial_health_score",
//     label: "Financial Health Score (Creditworthiness Input)",
//     type: "number",
//     category: "Engineered/Derived Features",
//   },
//   {
//     name: "total_balance",
//     label: "Total Balance (Creditworthiness Input)",
//     type: "number",
//     category: "Engineered/Derived Features",
//   },
//   {
//     name: "avg_credit_util",
//     label: "Average Credit Utilization (Creditworthiness Input)",
//     type: "number",
//     category: "Engineered/Derived Features",
//   },
//   {
//     name: "loan_to_income_1",
//     label: "Loan to Income Ratio 1 (Creditworthiness Input)",
//     type: "number",
//     category: "Engineered/Derived Features",
//   },
//   {
//     name: "loan_to_income_2",
//     label: "Loan to Income Ratio 2 (Creditworthiness Input)",
//     type: "number",
//     category: "Engineered/Derived Features",
//   },

//   // Add engineered features that Location model creates if they are to be displayed as inputs
//   // or need to be sent explicitly (though backend FE should handle these)
//   // For now, these are not direct frontend inputs, but rather internal to the backend's pipelines
//   // 'total_credit_limit', 'total_loan_amount_location', 'total_emi_sum', 'total_repayment_sum',
//   // 'total_inquiries_count', 'total_loans_count', 'credit_utilization_ratio_location'
// ];

// // --- DEFAULT DATA FOR ALL MODELS (Used for initial form population) ---
// const DEFAULT_ALL_DATA = {
//   // Demographic
//   age: 30,
//   gender: "Male",
//   marital_status: "Single",
//   residence_ownership: "Rented",
//   city: "Bengaluru",
//   state: "KA",
//   pin: "560001",

//   // Device
//   device_model: "iPhone 13",
//   device_category: "Smartphone",
//   platform: "iOS",
//   device_manufacturer: "Apple",

//   // Financial/Behavioral (var_0 to var_73)
//   var_0: 1000,
//   var_1: 500,
//   var_2: 10000,
//   var_3: 15000,
//   var_4: 2000,
//   var_5: 20000,
//   var_6: 5000,
//   var_7: 7000,
//   var_8: 1000,
//   var_9: 500,
//   var_10: 9000,
//   var_11: 8000,
//   var_12: 12000,
//   var_13: 20000,
//   var_14: 15000,
//   var_15: 1,
//   var_16: 0,
//   var_17: 700,
//   var_18: 300,
//   var_19: 600,
//   var_20: 10000,
//   var_21: 900,
//   var_22: 50000,
//   var_23: 60000,
//   var_24: 8000,
//   var_25: 0,
//   var_26: 70000,
//   var_27: 80000,
//   var_28: 90000,
//   var_29: 100000,
//   var_30: 1500,
//   var_31: 12000,
//   var_32: 700,
//   var_33: 110000,
//   var_34: 1800,
//   var_35: 2200,
//   var_36: 15000,
//   var_37: 400,
//   var_38: 2500,
//   var_39: 18000,
//   var_40: 5,
//   var_41: 800,
//   var_42: 22000,
//   var_43: 900,
//   var_44: 120000,
//   var_45: 2,
//   var_46: 600,
//   var_47: 130000,
//   var_48: 500,
//   var_49: 600,
//   var_50: 700,
//   var_51: 400,
//   var_52: 800,
//   var_53: 10,
//   var_54: 3,
//   var_55: 900,
//   var_56: 300,
//   var_57: 12,
//   var_58: 1,
//   var_59: 1000,
//   var_60: 11,
//   var_61: 0,
//   var_62: 2,
//   var_63: 15,
//   var_64: 18,
//   var_65: 25000,
//   var_66: 20,
//   var_67: 1000,
//   var_68: 3000,
//   var_69: 1100,
//   var_70: 1200,
//   var_71: 0,
//   var_72: 28000,
//   var_73: 1300,
//   var_74: "Good Payment History",
//   var_75: "A",

//   // Engineered features (Creditworthiness inputs, or potential outputs from master model)
//   final_predicted_income: 60000,
//   financial_health_score: 0.75,
//   total_balance: 15000,
//   avg_credit_util: 0.3,
//   loan_to_income_1: 0.2,
//   loan_to_income_2: 0.15,
// };

// // --- UTILITY FUNCTIONS ---
// // Helper to group features by category for better UI organization
// const groupFeaturesByCategory = (featuresConfig) => {
//   return featuresConfig.reduce((acc, feature) => {
//     if (!acc[feature.category]) {
//       acc[feature.category] = [];
//     }
//     acc[feature.category].push(feature);
//     return acc;
//   }, {});
// };

// // Initializes form data with default values for the current model
// const initializeFormDataWithDefaults = () => {
//   const newFormData = {};
//   ALL_FEATURES_CONFIG.forEach((feature) => {
//     newFormData[feature.name] =
//       DEFAULT_ALL_DATA[feature.name] !== undefined
//         ? DEFAULT_ALL_DATA[feature.name]
//         : "";
//   });
//   return newFormData;
// };

// // --- React App Component ---
// function App() {
//   // We maintain a single set of form data for ALL possible inputs, as we send all to unified backend.
//   const [formData, setFormData] = useState(initializeFormDataWithDefaults());
//   const [selectedModel, setSelectedModel] = useState("behavioral"); // Default selected model
//   const [allPredictions, setAllPredictions] = useState(null); // Stores results from '/predict_all'
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   // Map model types to their respective frontend display configurations
//   // The actual features sent to backend are always ALL_FEATURES_CONFIG
//   const MODEL_DISPLAY_CONFIGS = {
//     behavioral: {
//       title: "Behavioral Income Prediction",
//       features: ALL_FEATURES_CONFIG.filter(
//         (f) =>
//           f.category.includes("Behavioral") ||
//           f.category.includes("Credit Score")
//       ),
//     },
//     demographic: {
//       title: "Demographic Income Prediction",
//       features: ALL_FEATURES_CONFIG.filter(
//         (f) =>
//           f.category.includes("Demographic") ||
//           f.category.includes("Location") ||
//           f.category.includes("Device") ||
//           f.category.includes("Financial") ||
//           f.category.includes("Credit Score")
//       ),
//     },
//     creditworthiness: {
//       title: "Creditworthiness Prediction",
//       features: ALL_FEATURES_CONFIG.filter(
//         (f) =>
//           f.category.includes("Engineered/Derived") ||
//           f.category.includes("Demographic") ||
//           f.category.includes("Location") ||
//           f.category.includes("Device") ||
//           f.category.includes("Credit Score")
//       ),
//     },
//     location: {
//       title: "Location-Based Income Prediction",
//       features: ALL_FEATURES_CONFIG.filter(
//         (f) =>
//           f.category.includes("Location") ||
//           f.category.includes("Demographic") ||
//           f.category.includes("Device") ||
//           f.category.includes("Financial") ||
//           f.category.includes("Credit Score")
//       ),
//     },
//     master: {
//       title: "Master Income Prediction",
//       features: ALL_FEATURES_CONFIG,
//     }, // Master uses all raw features
//   };

//   const currentDisplayFeaturesConfig =
//     MODEL_DISPLAY_CONFIGS[selectedModel].features;
//   const featuresByCategory = groupFeaturesByCategory(
//     currentDisplayFeaturesConfig
//   );

//   // Handle input changes
//   const handleChange = (e) => {
//     const { name, value } = e.target;
//     setFormData((prevData) => ({
//       ...prevData,
//       [name]: value,
//     }));
//   };

//   // Handle form submission
//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setLoading(true);
//     setAllPredictions(null);
//     setError(null);

//     const dataToSend = {};
//     ALL_FEATURES_CONFIG.forEach((feature) => {
//       if (feature.type === "number") {
//         dataToSend[feature.name] =
//           formData[feature.name] === ""
//             ? null
//             : parseFloat(formData[feature.name]);
//       } else {
//         dataToSend[feature.name] =
//           formData[feature.name] === "" ? null : formData[feature.name];
//       }
//     });

//     // The single endpoint for the unified backend
//     const endpoint = "http://127.0.0.1:5000/predict_all";

//     try {
//       const response = await fetch(endpoint, {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify(dataToSend),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         throw new Error(
//           errorData.error ||
//             `Error from unified backend prediction. Status: ${response.status}`
//         );
//       }

//       const result = await response.json();
//       setAllPredictions(result); // Store all results
//       console.log("Unified Prediction Results:", result);
//     } catch (err) {
//       console.error("Prediction error:", err);
//       setError(
//         err.message ||
//           "Failed to get prediction from unified backend. Please check console and ensure Flask backend is running."
//       );
//     } finally {
//       setLoading(false);
//     }
//   };

//   // Handle reset button
//   const handleReset = () => {
//     setFormData(initializeFormDataWithDefaults()); // Reset to default values
//     setAllPredictions(null); // Clear all previous predictions
//     setError(null); // Clear any error messages
//   };

//   // Function to get the appropriate display class for prediction result (colors)
//   const getPredictionDisplayClass = (modelName, predictionValue) => {
//     if (
//       modelName === "creditworthiness" &&
//       typeof predictionValue === "string"
//     ) {
//       if (predictionValue === "Good") return "text-green-700";
//       if (predictionValue === "Average") return "text-yellow-700";
//       if (predictionValue === "Poor") return "text-red-700";
//     }
//     return "text-green-700"; // Default for income predictions
//   };

//   return (
//     <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4 font-sans">
//       <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-5xl border border-gray-200 my-8">
//         <h1 className="text-4xl font-extrabold text-gray-800 mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-700 pb-2">
//           Unified Income & Creditworthiness Predictor
//         </h1>

//         {/* Model Selection Buttons */}
//         <div className="flex flex-wrap justify-center mb-6 gap-3">
//           {Object.keys(MODEL_DISPLAY_CONFIGS).map((modelKey) => (
//             <button
//               key={modelKey}
//               onClick={() => setSelectedModel(modelKey)}
//               className={`px-5 py-2 rounded-full text-md font-semibold transition duration-300 ease-in-out shadow-md
//                                 ${
//                                   selectedModel === modelKey
//                                     ? (modelKey === "behavioral"
//                                         ? "bg-blue-600 shadow-blue-400/50"
//                                         : modelKey === "demographic"
//                                         ? "bg-purple-600 shadow-purple-400/50"
//                                         : modelKey === "creditworthiness"
//                                         ? "bg-red-600 shadow-red-400/50"
//                                         : modelKey === "location"
//                                         ? "bg-green-600 shadow-green-400/50"
//                                         : "bg-indigo-600 shadow-indigo-400/50") +
//                                       " text-white transform scale-105"
//                                     : "bg-gray-200 text-gray-700 hover:bg-gray-300"
//                                 }`}
//             >
//               {MODEL_DISPLAY_CONFIGS[modelKey].title.replace(" Prediction", "")}
//             </button>
//           ))}
//         </div>

//         <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
//           {MODEL_DISPLAY_CONFIGS[selectedModel].title}
//         </h2>

//         <form onSubmit={handleSubmit} className="space-y-6">
//           {Object.keys(featuresByCategory).map((category) => (
//             <div
//               key={category}
//               className="mb-8 p-6 bg-blue-50 rounded-lg shadow-inner border border-blue-200"
//             >
//               <h3 className="text-2xl font-bold text-blue-800 mb-5 border-b-2 border-blue-300 pb-3">
//                 {category}
//               </h3>
//               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
//                 {featuresByCategory[category].map((feature) => (
//                   <div key={feature.name} className="flex flex-col">
//                     <label
//                       htmlFor={feature.name}
//                       className="text-sm font-medium text-gray-700 mb-1"
//                     >
//                       {feature.label}
//                     </label>
//                     {feature.type === "select" ? (
//                       <select
//                         id={feature.name}
//                         name={feature.name}
//                         value={formData[feature.name] || ""}
//                         onChange={handleChange}
//                         className="p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out shadow-sm"
//                       >
//                         <option value="">Select...</option>
//                         {feature.options.map((option) => (
//                           <option key={option} value={option}>
//                             {option}
//                           </option>
//                         ))}
//                       </select>
//                     ) : (
//                       <input
//                         type={feature.type}
//                         id={feature.name}
//                         name={feature.name}
//                         value={formData[feature.name] || ""}
//                         onChange={handleChange}
//                         step={feature.type === "number" ? "any" : undefined}
//                         className="p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out shadow-sm"
//                         placeholder={`Enter ${feature.label.toLowerCase()}`}
//                       />
//                     )}
//                   </div>
//                 ))}
//               </div>
//             </div>
//           ))}

//           <div className="flex justify-center mt-8 space-x-4">
//             <button
//               type="submit"
//               className={`flex-1 px-8 py-4 bg-gradient-to-r from-green-500 to-teal-600 text-white font-bold text-lg rounded-xl shadow-lg hover:from-green-600 hover:to-teal-700 transition duration-300 ease-in-out transform hover:scale-105
//                                 ${
//                                   loading ? "opacity-60 cursor-not-allowed" : ""
//                                 }`}
//               disabled={loading}
//             >
//               {loading ? (
//                 <span className="flex items-center justify-center">
//                   <svg
//                     className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
//                     xmlns="http://www.w3.org/2000/svg"
//                     fill="none"
//                     viewBox="0 0 24 24"
//                   >
//                     <circle
//                       className="opacity-25"
//                       cx="12"
//                       cy="12"
//                       r="10"
//                       stroke="currentColor"
//                       strokeWidth="4"
//                     ></circle>
//                     <path
//                       className="opacity-75"
//                       fill="currentColor"
//                       d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
//                     ></path>
//                   </svg>
//                   Predicting...
//                 </span>
//               ) : (
//                 "Get All Predictions"
//               )}
//             </button>
//             <button
//               type="button"
//               onClick={handleReset}
//               className="flex-1 px-8 py-4 bg-gray-300 text-gray-800 font-bold text-lg rounded-xl shadow-lg hover:bg-gray-400 transition duration-300 ease-in-out transform hover:scale-105"
//             >
//               Reset Form
//             </button>
//           </div>
//         </form>

//         {/* Unified Prediction Results Display */}
//         {allPredictions && (
//           <div className="mt-10 p-6 bg-green-100 rounded-xl shadow-inner border border-green-300 text-center">
//             <h2 className="text-3xl font-semibold text-green-800 mb-3">
//               All Model Predictions:
//             </h2>

//             {allPredictions.master_income_prediction_error ? (
//               <p className="text-red-700 text-lg">
//                 Master Income Prediction:{" "}
//                 {allPredictions.master_income_prediction_error}
//               </p>
//             ) : allPredictions.master_income_prediction !== undefined ? (
//               <p className="text-xl font-bold text-green-700">
//                 Master Income: $
//                 {allPredictions.master_income_prediction.toLocaleString(
//                   "en-US",
//                   { minimumFractionDigits: 2, maximumFractionDigits: 2 }
//                 )}
//               </p>
//             ) : (
//               <p className="text-gray-600">
//                 Master Income Prediction: Not available
//               </p>
//             )}

//             {allPredictions.behavioral_income_prediction_error ? (
//               <p className="text-red-700 text-lg">
//                 Behavioral Income Prediction:{" "}
//                 {allPredictions.behavioral_income_prediction_error}
//               </p>
//             ) : allPredictions.behavioral_income_prediction !== undefined ? (
//               <p className="text-xl font-bold text-blue-700 mt-2">
//                 Behavioral Income: $
//                 {allPredictions.behavioral_income_prediction.toLocaleString(
//                   "en-US",
//                   { minimumFractionDigits: 2, maximumFractionDigits: 2 }
//                 )}
//               </p>
//             ) : (
//               <p className="text-gray-600">
//                 Behavioral Income Prediction: Not available
//               </p>
//             )}

//             {allPredictions.demographic_income_prediction_error ? (
//               <p className="text-red-700 text-lg">
//                 Demographic Income Prediction:{" "}
//                 {allPredictions.demographic_income_prediction_error}
//               </p>
//             ) : allPredictions.demographic_income_prediction !== undefined ? (
//               <p className="text-xl font-bold text-purple-700 mt-2">
//                 Demographic Income: $
//                 {allPredictions.demographic_income_prediction.toLocaleString(
//                   "en-US",
//                   { minimumFractionDigits: 2, maximumFractionDigits: 2 }
//                 )}
//               </p>
//             ) : (
//               <p className="text-gray-600">
//                 Demographic Income Prediction: Not available
//               </p>
//             )}

//             {allPredictions.location_income_prediction_error ? (
//               <p className="text-red-700 text-lg">
//                 Location Income Prediction:{" "}
//                 {allPredictions.location_income_prediction_error}
//               </p>
//             ) : allPredictions.location_income_prediction !== undefined ? (
//               <p className="text-xl font-bold text-green-800 mt-2">
//                 Location Income: $
//                 {allPredictions.location_income_prediction.toLocaleString(
//                   "en-US",
//                   { minimumFractionDigits: 2, maximumFractionDigits: 2 }
//                 )}
//               </p>
//             ) : (
//               <p className="text-gray-600">
//                 Location Income Prediction: Not available
//               </p>
//             )}

//             {allPredictions.creditworthiness_prediction_error ? (
//               <p className="text-red-700 text-lg">
//                 Creditworthiness Prediction:{" "}
//                 {allPredictions.creditworthiness_prediction_error}
//               </p>
//             ) : allPredictions.creditworthiness_prediction_label !==
//               undefined ? (
//               <div className="mt-4">
//                 <p className="text-xl font-bold text-red-700">
//                   Creditworthiness:{" "}
//                   <span
//                     className={getPredictionDisplayClass(
//                       "creditworthiness",
//                       allPredictions.creditworthiness_prediction_label
//                     )}
//                   >
//                     {allPredictions.creditworthiness_prediction_label}
//                   </span>
//                 </p>
//                 {allPredictions.creditworthiness_prediction_probabilities && (
//                   <div className="mt-2 text-left inline-block">
//                     <h3 className="text-lg font-medium text-gray-700 mb-1">
//                       Probabilities:
//                     </h3>
//                     <ul className="list-disc list-inside">
//                       {Object.entries(
//                         allPredictions.creditworthiness_prediction_probabilities
//                       ).map(([label, prob]) => (
//                         <li key={label} className="text-gray-600">
//                           {label}: {(prob * 100).toFixed(2)}%
//                         </li>
//                       ))}
//                     </ul>
//                   </div>
//                 )}
//               </div>
//             ) : (
//               <p className="text-gray-600">
//                 Creditworthiness Prediction: Not available
//               </p>
//             )}

//             <p className="text-gray-600 mt-4">
//               All predictions are based on the full set of inputs provided.
//             </p>
//           </div>
//         )}

//         {error && (
//           <div className="mt-10 p-6 bg-red-100 rounded-xl shadow-inner border border-red-300 text-center">
//             <h2 className="text-xl font-semibold text-red-800 mb-3">Error:</h2>
//             <p className="text-red-700">{error}</p>
//             <p className="text-gray-600 mt-2">
//               Please ensure the Flask backend is running on
//               `http://127.0.0.1:5000`.
//             </p>
//           </div>
//         )}

//         <div className="mt-8 text-center text-gray-500 text-sm">
//           <p>
//             This application uses pre-trained LightGBM and Random Forest models
//             for predictions, served by a unified Flask backend.
//           </p>
//           <p>Developed with React and Flask.</p>
//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;

// script.js (Vanilla JS for Unified Frontend)

// --- CONFIGURATION FOR ALL POSSIBLE INPUT FEATURES ACROSS ALL MODELS ---
// This comprehensive list is used to build the single payload sent to the unified backend.
const ALL_FEATURES_CONFIG = [
  // Demographic Information
  {
    name: "age",
    label: "Age (Numerical)",
    type: "number",
    category: "Demographic Information",
  },
  {
    name: "gender",
    label: "Gender (Categorical)",
    type: "select",
    options: ["Male", "Female", "Other"],
    category: "Demographic Information",
  },
  {
    name: "marital_status",
    label: "Marital Status (Categorical)",
    type: "select",
    options: ["Married", "Single", "Divorced", "Widowed"],
    category: "Demographic Information",
  },
  {
    name: "residence_ownership",
    label: "Residence Ownership (Categorical)",
    type: "select",
    options: ["Owned", "Rented", "Jointly Owned"],
    category: "Demographic Information",
  },
  {
    name: "city",
    label: "City (Categorical)",
    type: "text",
    category: "Location Information",
  },
  {
    name: "state",
    label: "State (Categorical)",
    type: "text",
    category: "Location Information",
  },
  {
    name: "pin",
    label: "PIN Code (Categorical)",
    type: "text",
    category: "Location Information",
  },

  // Device Information
  {
    name: "device_model",
    label: "Device Model (Categorical)",
    type: "text",
    category: "Device Information",
  },
  {
    name: "device_category",
    label: "Device Category (Categorical)",
    type: "select",
    options: ["Smartphone", "Tablet", "Laptop", "Desktop"],
    category: "Device Information",
  },
  {
    name: "platform",
    label: "Platform (Categorical)",
    type: "select",
    options: ["iOS", "Android", "Windows", "macOS", "Linux"],
    category: "Device Information",
  },
  {
    name: "device_manufacturer",
    label: "Device Manufacturer (Categorical)",
    type: "text",
    category: "Device Information",
  },

  // Financial/Behavioral Metrics (var_0 to var_73) - comprehensive list
  {
    name: "var_0",
    label: "Var 0 (Balance 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_1",
    label: "Var 1 (Balance 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_2",
    label: "Var 2 (Credit Limit 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_3",
    label: "Var 3 (Credit Limit 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_4",
    label: "Var 4 (Balance 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_5",
    label: "Var 5 (Credit Limit 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_6",
    label: "Var 6 (Loan Amt 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_7",
    label: "Var 7 (Loan Amt 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_8",
    label: "Var 8 (Business Balance)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_9",
    label: "Var 9 (Total EMI 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_10",
    label: "Var 10 (Active Credit Limit 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_11",
    label: "Var 11 (Credit Limit Recent 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_12",
    label: "Var 12 (Credit Limit 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_13",
    label: "Var 13 (Loan Amt Large Tenure)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_14",
    label: "Var 14 (Primary Loan Amt)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_15",
    label: "Var 15 (Total Inquiries 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_16",
    label: "Var 16 (Total Inquiries 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_17",
    label: "Var 17 (Total EMI 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_18",
    label: "Var 18 (Balance 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_19",
    label: "Var 19 (Balance 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_20",
    label: "Var 20 (Loan Amt 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_21",
    label: "Var 21 (Balance 6)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_22",
    label: "Var 22 (Credit Limit 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_23",
    label: "Var 23 (Credit Limit 6)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_24",
    label: "Var 24 (Loan Amt Recent)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_25",
    label: "Var 25 (Total Inquiries Recent)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_26",
    label: "Var 26 (Credit Limit 7)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_27",
    label: "Var 27 (Credit Limit 8)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_28",
    label: "Var 28 (Credit Limit 9)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_29",
    label: "Var 29 (Credit Limit 10)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_30",
    label: "Var 30 (Balance 7)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_31",
    label: "Var 31 (Loan Amt 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_32",
    label: "Var 32 (Credit Score)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_33",
    label: "Var 33 (Credit Limit 11)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_34",
    label: "Var 34 (Balance 8)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_35",
    label: "Var 35 (Balance 9)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_36",
    label: "Var 36 (Loan Amt 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_37",
    label: "Var 37 (Repayment 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_38",
    label: "Var 38 (Balance 10)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_39",
    label: "Var 39 (Loan Amt 6)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_40",
    label: "Var 40 (Closed Loan)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_41",
    label: "Var 41 (Total EMI 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_42",
    label: "Var 42 (Loan Amt 7)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_43",
    label: "Var 43 (Total EMI 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_44",
    label: "Var 44 (Credit Limit 12)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_45",
    label: "Var 45 (Total Inquiries 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_46",
    label: "Var 46 (Total EMI 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_47",
    label: "Var 47 (Credit Limit 13)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_48",
    label: "Var 48 (Repayment 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_49",
    label: "Var 49 (Repayment 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_50",
    label: "Var 50 (Repayment 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_51",
    label: "Var 51 (Total EMI 6)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_52",
    label: "Var 52 (Repayment 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_53",
    label: "Var 53 (Total Loans 1)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_54",
    label: "Var 54 (Closed Total Loans)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_55",
    label: "Var 55 (Repayment 6)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_56",
    label: "Var 56 (Total EMI 7)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_57",
    label: "Var 57 (Total Loans 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_58",
    label: "Var 58 (Total Inquiries 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_59",
    label: "Var 59 (Balance 11)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_60",
    label: "Var 60 (Total Loans 2)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_61",
    label: "Var 61 (Total Inquiries 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_62",
    label: "Var 62 (Total Loan Recent)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_63",
    label: "Var 63 (Total Loans 3)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_64",
    label: "Var 64 (Total Loans 4)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_65",
    label: "Var 65 (Loan Amt 8)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_66",
    label: "Var 66 (Total Loans 5)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_67",
    label: "Var 67 (Repayment 7)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_68",
    label: "Var 68 (Balance 12)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_69",
    label: "Var 69 (Repayment 8)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_70",
    label: "Var 70 (Repayment 9)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_71",
    label: "Var 71 (Total Inquiries 6)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_72",
    label: "Var 72 (Loan Amt 9)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_73",
    label: "Var 73 (Repayment 10)",
    type: "number",
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_74",
    label: "Var 74 (Score Comments)",
    type: "select",
    options: [
      "Good Payment History",
      "Late Payments",
      "High Credit Utilization",
      "Low Credit Score",
      "No Credit History",
      "Bankruptcies",
      "Defaults",
    ],
    category: "Financial/Behavioral Metrics",
  },
  {
    name: "var_75",
    label: "Var 75 (Score Type)",
    type: "select",
    options: ["A", "B", "C", "D", "E", "F"],
    category: "Financial/Behavioral Metrics",
  },

  // Engineered Features expected by Creditworthiness (these need to be provided as inputs to backend)
  // In a full, chained system, 'final_predicted_income' would come from the Master Model's output.
  // For this unified form, they are manual inputs to demonstrate the full set of parameters.
  {
    name: "final_predicted_income",
    label: "Final Predicted Income (for Creditworthiness)",
    type: "number",
    category: "Creditworthiness Specific Inputs",
  },
  {
    name: "financial_health_score",
    label: "Financial Health Score (for Creditworthiness)",
    type: "number",
    category: "Creditworthiness Specific Inputs",
  },
  {
    name: "total_balance",
    label: "Total Balance (for Creditworthiness)",
    type: "number",
    category: "Creditworthiness Specific Inputs",
  },
  {
    name: "avg_credit_util",
    label: "Average Credit Utilization (for Creditworthiness)",
    type: "number",
    category: "Creditworthiness Specific Inputs",
  },
  {
    name: "loan_to_income_1",
    label: "Loan to Income Ratio 1 (for Creditworthiness)",
    type: "number",
    category: "Creditworthiness Specific Inputs",
  },
  {
    name: "loan_to_income_2",
    label: "Loan to Income Ratio 2 (for Creditworthiness)",
    type: "number",
    category: "Creditworthiness Specific Inputs",
  },
];

// --- DEFAULT DATA FOR ALL MODELS (Used for initial form population) ---
// These values are arbitrary examples and should be replaced with meaningful defaults
// or values typical for your expected input range.
const DEFAULT_ALL_DATA = {
  // Demographic
  age: 30,
  gender: "Male",
  marital_status: "Single",
  residence_ownership: "Rented",
  city: "Bengaluru",
  state: "KA",
  pin: "560001",

  // Device
  device_model: "iPhone 13",
  device_category: "Smartphone",
  platform: "iOS",
  device_manufacturer: "Apple",

  // Financial/Behavioral (var_0 to var_75)
  var_0: 1000,
  var_1: 500,
  var_2: 10000,
  var_3: 15000,
  var_4: 2000,
  var_5: 20000,
  var_6: 5000,
  var_7: 7000,
  var_8: 1000,
  var_9: 500,
  var_10: 9000,
  var_11: 8000,
  var_12: 12000,
  var_13: 20000,
  var_14: 15000,
  var_15: 1,
  var_16: 0,
  var_17: 700,
  var_18: 300,
  var_19: 600,
  var_20: 10000,
  var_21: 900,
  var_22: 50000,
  var_23: 60000,
  var_24: 8000,
  var_25: 0,
  var_26: 70000,
  var_27: 80000,
  var_28: 90000,
  var_29: 100000,
  var_30: 1500,
  var_31: 12000,
  var_32: 700,
  var_33: 110000,
  var_34: 1800,
  var_35: 2200,
  var_36: 15000,
  var_37: 400,
  var_38: 2500,
  var_39: 18000,
  var_40: 5,
  var_41: 800,
  var_42: 22000,
  var_43: 900,
  var_44: 120000,
  var_45: 2,
  var_46: 600,
  var_47: 130000,
  var_48: 500,
  var_49: 600,
  var_50: 700,
  var_51: 400,
  var_52: 800,
  var_53: 10,
  var_54: 3,
  var_55: 900,
  var_56: 300,
  var_57: 12,
  var_58: 1,
  var_59: 1000,
  var_60: 11,
  var_61: 0,
  var_62: 2,
  var_63: 15,
  var_64: 18,
  var_65: 25000,
  var_66: 20,
  var_67: 1000,
  var_68: 3000,
  var_69: 1100,
  var_70: 1200,
  var_71: 0,
  var_72: 28000,
  var_73: 1300,
  var_74: "Good Payment History",
  var_75: "A",

  // Engineered features (placeholder values for manual input)
  final_predicted_income: 60000,
  financial_health_score: 0.75,
  total_balance: 15000,
  avg_credit_util: 0.3,
  loan_to_income_1: 0.2,
  loan_to_income_2: 0.15,
};

// --- DOM ELEMENTS ---
const formInputsContainer = document.getElementById("form-inputs-container");
const predictionForm = document.getElementById("prediction-form");
const submitButton = document.getElementById("submit-button");
const resetButton = document.getElementById("reset-button");
const predictionResultsContainer = document.getElementById(
  "prediction-results-container"
);
const predictionOutputDetails = document.getElementById(
  "prediction-output-details"
);
const errorMessageContainer = document.getElementById(
  "error-message-container"
);
const errorText = document.getElementById("error-text");

// --- DATA STATE (in a simple JS object) ---
let formData = {};

// --- UTILITY FUNCTIONS ---

// Groups features by category for organized display
const groupFeaturesByCategory = (featuresConfig) => {
  return featuresConfig.reduce((acc, feature) => {
    if (!acc[feature.category]) {
      acc[feature.category] = [];
    }
    acc[feature.category].push(feature);
    return acc;
  }, {});
};

// Renders the form inputs based on ALL_FEATURES_CONFIG
const renderFormInputs = () => {
  formInputsContainer.innerHTML = ""; // Clear previous inputs
  const featuresByCategory = groupFeaturesByCategory(ALL_FEATURES_CONFIG);

  for (const category in featuresByCategory) {
    const categoryDiv = document.createElement("div");
    categoryDiv.className =
      "mb-8 p-6 bg-blue-50 rounded-lg shadow-inner border border-blue-200";

    const categoryTitle = document.createElement("h3");
    categoryTitle.className =
      "text-2xl font-bold text-blue-800 mb-5 border-b-2 border-blue-300 pb-3";
    categoryTitle.textContent = category;
    categoryDiv.appendChild(categoryTitle);

    const gridDiv = document.createElement("div");
    gridDiv.className = "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6";

    featuresByCategory[category].forEach((feature) => {
      const fieldDiv = document.createElement("div");
      fieldDiv.className = "flex flex-col";

      const label = document.createElement("label");
      label.htmlFor = feature.name;
      label.className = "text-sm font-medium text-gray-700 mb-1";
      label.textContent = feature.label;
      fieldDiv.appendChild(label);

      let inputElement;
      if (feature.type === "select") {
        inputElement = document.createElement("select");
        inputElement.className =
          "p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out shadow-sm";
        inputElement.id = feature.name;
        inputElement.name = feature.name;

        const defaultOption = document.createElement("option");
        defaultOption.value = "";
        defaultOption.textContent = "Select...";
        inputElement.appendChild(defaultOption);

        feature.options.forEach((option) => {
          const optionElement = document.createElement("option");
          optionElement.value = option;
          optionElement.textContent = option;
          inputElement.appendChild(optionElement);
        });
      } else {
        inputElement = document.createElement("input");
        inputElement.type = feature.type;
        inputElement.id = feature.name;
        inputElement.name = feature.name;
        inputElement.className =
          "p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out shadow-sm";
        inputElement.placeholder = `Enter ${feature.label.toLowerCase()}`;
        if (feature.type === "number") {
          inputElement.step = "any";
        }
      }

      // Set initial value from formData or empty string
      inputElement.value =
        formData[feature.name] !== undefined ? formData[feature.name] : "";
      inputElement.addEventListener("input", handleInputChange);
      fieldDiv.appendChild(inputElement);
      gridDiv.appendChild(fieldDiv);
    });
    categoryDiv.appendChild(gridDiv);
    formInputsContainer.appendChild(categoryDiv);
  }
};

// Initializes form data with default values
const initializeFormDataWithDefaults = () => {
  formData = {}; // Clear existing data
  ALL_FEATURES_CONFIG.forEach((feature) => {
    formData[feature.name] =
      DEFAULT_ALL_DATA[feature.name] !== undefined
        ? DEFAULT_ALL_DATA[feature.name]
        : "";
  });
  renderFormInputs(); // Re-render form with defaults
};

// Handles input changes and updates formData object
const handleInputChange = (event) => {
  const { name, value, type } = event.target;
  if (type === "number") {
    formData[name] = value === "" ? "" : parseFloat(value);
  } else {
    formData[name] = value;
  }
};

// Displays all predictions from the backend response
const displayAllPredictions = (predictions) => {
  predictionOutputDetails.innerHTML = ""; // Clear previous results
  predictionResultsContainer.classList.remove("hidden"); // Show results container
  errorMessageContainer.classList.add("hidden"); // Hide any error messages

  // Master Income Prediction
  if (predictions.master_income_prediction_error) {
    addPredictionError(
      "Master Income Prediction",
      predictions.master_income_prediction_error
    );
  } else if (predictions.master_income_prediction !== undefined) {
    addPredictionResult(
      "Master Income Prediction",
      `$${predictions.master_income_prediction.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      "text-indigo-700"
    );
  } else {
    addPredictionStatus("Master Income Prediction", "Not available.");
  }

  // Behavioral Income Prediction
  if (predictions.behavioral_income_prediction_error) {
    addPredictionError(
      "Behavioral Income Prediction",
      predictions.behavioral_income_prediction_error
    );
  } else if (predictions.behavioral_income_prediction !== undefined) {
    addPredictionResult(
      "Behavioral Income Prediction",
      `$${predictions.behavioral_income_prediction.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      "text-blue-700"
    );
  } else {
    addPredictionStatus("Behavioral Income Prediction", "Not available.");
  }

  // Demographic Income Prediction
  if (predictions.demographic_income_prediction_error) {
    addPredictionError(
      "Demographic Income Prediction",
      predictions.demographic_income_prediction_error
    );
  } else if (predictions.demographic_income_prediction !== undefined) {
    addPredictionResult(
      "Demographic Income Prediction",
      `$${predictions.demographic_income_prediction.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      "text-purple-700"
    );
  } else {
    addPredictionStatus("Demographic Income Prediction", "Not available.");
  }

  // Location Income Prediction
  if (predictions.location_income_prediction_error) {
    addPredictionError(
      "Location Income Prediction",
      predictions.location_income_prediction_error
    );
  } else if (predictions.location_income_prediction !== undefined) {
    addPredictionResult(
      "Location Income Prediction",
      `$${predictions.location_income_prediction.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      "text-green-700"
    );
  } else {
    addPredictionStatus("Location Income Prediction", "Not available.");
  }

  // Device Income Prediction
  if (predictions.device_income_prediction_error) {
    addPredictionError(
      "Device Income Prediction",
      predictions.device_income_prediction_error
    );
  } else if (predictions.device_income_prediction !== undefined) {
    addPredictionResult(
      "Device Income Prediction",
      `$${predictions.device_income_prediction.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      "text-orange-700" // Using orange for device model
    );
  } else {
    addPredictionStatus("Device Income Prediction", "Not available.");
  }

  // Creditworthiness Prediction
  if (predictions.creditworthiness_prediction_error) {
    addPredictionError(
      "Creditworthiness Prediction",
      predictions.creditworthiness_prediction_error
    );
  } else if (predictions.creditworthiness_prediction_label !== undefined) {
    let labelClass = "text-gray-800"; // Default
    if (predictions.creditworthiness_prediction_label === "Good")
      labelClass = "text-green-700";
    else if (predictions.creditworthiness_prediction_label === "Average")
      labelClass = "text-yellow-700";
    else if (predictions.creditworthiness_prediction_label === "Poor")
      labelClass = "text-red-700";

    const creditworthinessDiv = document.createElement("div");
    creditworthinessDiv.className =
      "p-2 border-b border-gray-200 last:border-b-0";
    creditworthinessDiv.innerHTML = `
            <p class="text-xl font-bold text-gray-800">
                Creditworthiness: <span class="${labelClass}">${predictions.creditworthiness_prediction_label}</span>
            </p>
        `;
    if (predictions.creditworthiness_prediction_probabilities) {
      const probsList = document.createElement("ul");
      probsList.className =
        "list-disc list-inside text-gray-600 mt-2 text-left mx-auto max-w-sm";
      for (const [label, prob] of Object.entries(
        predictions.creditworthiness_prediction_probabilities
      )) {
        const listItem = document.createElement("li");
        listItem.textContent = `${label}: ${(prob * 100).toFixed(2)}%`;
        probsList.appendChild(listItem);
      }
      creditworthinessDiv.appendChild(probsList);
    }
    predictionOutputDetails.appendChild(creditworthinessDiv);
  } else {
    addPredictionStatus("Creditworthiness Prediction", "Not available.");
  }
};

// Helper for adding a single prediction result
const addPredictionResult = (title, value, colorClass) => {
  const div = document.createElement("div");
  div.className = "p-2 border-b border-gray-200 last:border-b-0";
  div.innerHTML = `
        <p class="text-lg font-semibold text-gray-800">${title}:</p>
        <p class="text-3xl font-extrabold ${colorClass}">${value}</p>
    `;
  predictionOutputDetails.appendChild(div);
};

// Helper for adding a model status if prediction wasn't run
const addPredictionStatus = (title, message) => {
  const div = document.createElement("div");
  div.className =
    "p-2 border-b border-gray-200 last:border-b-0 text-gray-600 italic";
  div.innerHTML = `
        <p class="text-lg font-semibold text-gray-800">${title}:</p>
        <p>${message}</p>
    `;
  predictionOutputDetails.appendChild(div);
};

// Helper for adding a model error
const addPredictionError = (title, errorMessage) => {
  const div = document.createElement("div");
  div.className = "p-2 border-b border-red-200 last:border-b-0 text-red-700";
  div.innerHTML = `
        <p class="text-lg font-semibold text-red-800">${title}:</p>
        <p>Error: ${errorMessage}</p>
    `;
  predictionOutputDetails.appendChild(div);
};

// Displays error message
const displayError = (message) => {
  errorMessageContainer.classList.remove("hidden");
  predictionResultsContainer.classList.add("hidden"); // Hide results
  errorText.textContent = message;
  submitButton.classList.remove("opacity-60", "cursor-not-allowed");
  submitButton.innerHTML = "Get All Predictions";
};

// Hides error message
const hideError = () => {
  errorMessageContainer.classList.add("hidden");
  errorText.textContent = "";
};

// --- EVENT HANDLERS ---

const handleSubmit = async (event) => {
  event.preventDefault();
  submitButton.classList.add("opacity-60", "cursor-not-allowed");
  submitButton.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg> Predicting...`;
  predictionResultsContainer.classList.add("hidden"); // Hide previous results
  hideError(); // Hide previous errors

  const dataToSend = {};
  ALL_FEATURES_CONFIG.forEach((feature) => {
    const value = formData[feature.name];
    if (feature.type === "number") {
      dataToSend[feature.name] = value === "" ? null : parseFloat(value);
    } else {
      dataToSend[feature.name] = value === "" ? null : value;
    }
  });

  const endpoint = "http://127.0.0.1:5000/predict_all"; // Unified endpoint

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(dataToSend),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.error || `Error from backend. Status: ${response.status}`
      );
    }

    const result = await response.json();
    console.log("Unified Prediction Results:", result);
    displayAllPredictions(result); // Display all results
  } catch (err) {
    console.error("Prediction error:", err);
    displayError(
      err.message ||
        "Failed to get predictions. Please check console and ensure Flask backend is running."
    );
  } finally {
    submitButton.classList.remove("opacity-60", "cursor-not-allowed");
    submitButton.innerHTML = "Get All Predictions";
  }
};

const handleReset = () => {
  initializeFormDataWithDefaults(); // Reset form fields to defaults
  predictionResultsContainer.classList.add("hidden"); // Hide results
  hideError(); // Hide error
};

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
  initializeFormDataWithDefaults(); // Populate form with defaults on page load

  // Attach event listeners
  predictionForm.addEventListener("submit", handleSubmit);
  resetButton.addEventListener("click", handleReset);
});
