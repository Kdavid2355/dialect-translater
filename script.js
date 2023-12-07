let isStandardToDialect = true;

async function query(data, isStandardToDialect) {
    try {
        const apiUrl = isStandardToDialect 
            ? "https://api-inference.huggingface.co/models/sswoo123/t5_dialect"
            : "https://api-inference.huggingface.co/models/sswoo123/t5_reverse_dialect";
        const response = await fetch(apiUrl, {
            headers: { Authorization: "Bearer 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" },
            method: "POST",
            body: JSON.stringify(data),
        });
        const result = await response.json();
        return result; } catch(error){
            console.error("API 요청오류:", error);
            return null;
        }
}

function updateUIForModeChange() {
    console.log("모드 전환:", isStandardToDialect); 
    const title = isStandardToDialect ? "표준어->방언 번역기(제주도)" : "방언->표준어 번역기(제주도)";
    document.getElementById('title').textContent = title;
    document.getElementById("inputText").placeholder = isStandardToDialect ? "표준어를 입력해주세요!" : "방언을 입력해주세요!";
    document.getElementById("outputText").placeholder = isStandardToDialect ? "여기에 방언이 출력됩니다!" : "여기에 표준어가 출력됩니다!";
    // Clear existing text in textareas
    document.getElementById("inputText").value = '';
    document.getElementById("outputText").value = '';
}

document.getElementById("toggleModeButton").addEventListener("click", () => {
    isStandardToDialect = !isStandardToDialect;
    updateUIForModeChange();
});

document.getElementById("translateButton").addEventListener("click", async () => {
    console.log("번역 시작", isStandardToDialect); // 모드 확인 로그
    const inputText = document.getElementById("inputText").value;
    console.log("입력 텍스트:", inputText); // 입력된 텍스트 로그
    const response = await query({ "inputs": inputText }, isStandardToDialect);
    console.log("API 응답:", response); // API 응답 로그
    const generatedText = response[0]?.generated_text || "No response";
    console.log("생성된 텍스트:", generatedText); // 생성된 텍스트 로그
    document.getElementById("outputText").value = generatedText;
});
