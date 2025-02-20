require('dotenv').config();
const puppeteer = require('puppeteer');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Relevant roles for passing CVs
const relevantRoles = ["Software Engineer", "Software Developer"];

// Output directory setup
const outputDir = path.join(__dirname, "data/generated_train_CVs");
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Generate job titles & classifications
function generateRandomJobTitle() {
    if (Math.random() < 0.7) {
        return {
            jobTitle: getRandomElement(relevantRoles),
            classification: "Pass"
        };
    } else {
        return {
            jobTitle: generateRandomIrrelevantJob(),
            classification: "Fail"
        };
    }
}

// Generate random irrelevant jobs
function generateRandomIrrelevantJob() {
    const randomJobs = [
        "Psychologist", "Mechanical Engineer", "Biomedical Engineer",
        "Architect", "Data Scientist", "Data Analyst", "Sales Executive", "Data Engineer"
    ];
    return getRandomElement(randomJobs);
}

// Select a random item from an array
function getRandomElement(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

// Generate a GPT-based structured CV prompt
function getCVPrompt(profile) {
    return `
Generate a highly detailed, well-structured, and realistic CV for the following:
- **Name:** ${profile.name}
- **Job Title:** ${profile.jobTitle}
- **Classification:** ${profile.classification} (Pass or Fail)

CV **Structure:**
1. **Contact Information** (Email, Phone, LinkedIn)
2. **Professional Summary** (3-4 sentences summarizing experience)
3. **Work Experience** (2-3 roles, detailed bullet points for each)
4. **Education** (Degree, University, Year)
5. **Skills** (Technical + Soft skills)
6. **Projects** (2-3 relevant projects with details)
7. **Certifications** (Relevant industry certifications)
8. **Hobbies & Interests** (Optional)
9. **References available upon request**

**Important:**  
- If **Pass**, make it well-structured, clear, and professional.
- If **Fail**, make it **poorly structured**, **messy**, **unrelated**, or missing important sections.
- Ensure **at least one full A4 page of content**.
- Output the CV in **HTML format** for PDF conversion.
`;
}

// CSV setup
const csvFilePath = path.join(outputDir, "cv_labels.csv");
if (!fs.existsSync(csvFilePath)) {
    fs.writeFileSync(csvFilePath, "CV_Name,Job_Title,Classification\n", "utf8");
}

// Append CV details to CSV
function appendToCSV(fileName, jobTitle, classification) {
    const csvEntry = `${fileName},${jobTitle},${classification}\n`;
    fs.appendFileSync(csvFilePath, csvEntry, "utf8");
}

// Main function to generate CVs
(async () => {
    const browser = await puppeteer.launch({
        headless: "new",
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
        timeout: 60000
    });

    for (let i = 1; i <= 200; i++) {
        const profileData = generateRandomJobTitle();
        const profile = {
            name: `Person_${i}`,
            jobTitle: profileData.jobTitle,
            classification: profileData.classification
        };

        const prompt = getCVPrompt(profile);
        const fileName = `cv_${i}.pdf`;
        const filePath = path.join(outputDir, fileName);

        try {
            const res = await openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [{ role: "user", content: prompt }],
                temperature: 0.7,
                max_tokens: 2000
            });

            const htmlContent = res.choices[0].message.content;

            const page = await browser.newPage();
            await page.setContent(htmlContent, { waitUntil: 'domcontentloaded' });
            await new Promise(resolve => setTimeout(resolve, 1000)); // Ensure rendering before PDF

            try {
                await page.pdf({ path: filePath, format: 'A4' });
                console.log(`✅ Generated ${profile.classification} CV ${i}: ${profile.name} - ${profile.jobTitle}`);

                // Save CV classification in CSV
                appendToCSV(fileName, profile.jobTitle, profile.classification);

            } catch (pdfError) {
                console.error(`❌ Error generating PDF for ${filePath}:`, pdfError);
            }

            await page.close();
        } catch (error) {
            console.error(`❌ Error generating CV ${i}:`, error);
        }

        // Introduce slight delay
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    await browser.close();
    console.log("🎉 All CVs have been generated successfully!");
})();
