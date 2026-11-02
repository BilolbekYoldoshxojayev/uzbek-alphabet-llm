// Batch converter: old Uzbek Latin (1995) -> 2026 alphabet, one line in, one line out.
//
//   dotnet run -c Release --project converter/YangiAlifbo.Cli -- <in.txt> <out.txt> [--words <decisions.tsv>]
//
// Unlike TextConverter, leading/trailing punctuation (“ « ‘ ( …) is peeled off each
// token before the engine sees it, so a quoted word like “Shahar” is still converted.
// --words writes every word that had a convertible pattern: line, word, result, changed.
using System.Text;
using System.Text.RegularExpressions;
using YangiAlifbo.Core;
using YangiAlifbo.Win32;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: <in.txt> <out.txt> [--words <decisions.tsv>]");
    return 2;
}

// The default detector also loads the user's personal whitelist/blacklist from AppData.
// Those would make results machine-dependent, so refuse to run unless they are empty.
foreach (var path in new[] { LanguageDetector.GetWhitelistPath(), LanguageDetector.GetBlacklistPath() })
{
    if (File.Exists(path) && File.ReadLines(path).Any(l => l.Trim().Length > 0 && !l.TrimStart().StartsWith('#')))
    {
        Console.Error.WriteLine($"error: user list is not empty: {path}");
        return 3;
    }
}

using var detector = new LanguageDetector();
string? wordsPath = args.Length >= 4 && args[2] == "--words" ? args[3] : null;
var decisions = wordsPath is null ? null : new StreamWriter(wordsPath, false, new UTF8Encoding(false));
decisions?.WriteLine("line\tword\tresult\tchanged");

var split = new Regex(@"(\s+)", RegexOptions.Compiled);
string apostrophes = "'‘’ʻʼ`´‛′";
int lineNo = 0, convertible = 0, changed = 0;
using var output = new StreamWriter(args[1], false, new UTF8Encoding(false));

foreach (string line in File.ReadLines(args[0], Encoding.UTF8))
{
    lineNo++;
    var sb = new StringBuilder(line.Length + 8);
    foreach (string token in split.Split(line))
    {
        if (token.Length == 0 || char.IsWhiteSpace(token[0])) { sb.Append(token); continue; }

        // Peel punctuation: a trailing apostrophe stays when it follows o/g (tog‘, bo‘).
        int start = 0, end = token.Length;
        while (start < end && !char.IsLetterOrDigit(token[start])) start++;
        while (end > start && !char.IsLetterOrDigit(token[end - 1]))
        {
            if (apostrophes.Contains(token[end - 1]) && end - 2 >= start && "oOgG".Contains(token[end - 2])) break;
            end--;
        }
        // The lexicon holds single words, so hyphenated forms (boʻlsa-da, video-oʻyin) and
        // inner quotes or slashes (yozish"ni, kelish/davom) are converted part by part.
        sb.Append(token, 0, start);
        foreach (string part in Regex.Split(token[start..end], "([-\"/])"))
        {
            string result = part;
            if (part.Length >= 2 && UzbekPhonetics.HasConvertiblePatterns(part))
            {
                result = WordConverter.Convert(detector, part, TransliterationMode.Strict, slang: false) ?? part;
                convertible++;
                if (result != part) changed++;
                decisions?.WriteLine($"{lineNo}\t{part}\t{result}\t{(result != part ? 1 : 0)}");
            }
            sb.Append(result);
        }
        sb.Append(token, end, token.Length - end);
    }
    output.WriteLine(sb.ToString());
}
decisions?.Dispose();
Console.Error.WriteLine($"lines={lineNo} convertible_words={convertible} converted={changed} kept={convertible - changed}");
return 0;
